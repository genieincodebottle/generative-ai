from dataclasses import dataclass
from typing import List, Tuple
from time import time
from datetime import datetime
import copy
import torch
from transformers import (
    BitsAndBytesConfig,
    AutoTokenizer,
    AutoModelForCausalLM,
    DynamicCache
)
from sentence_transformers import SentenceTransformer

# Allowlist the DynamicCache class
torch.serialization.add_safe_globals([DynamicCache])
torch.serialization.add_safe_globals([set])


class ModelLoadError(RuntimeError):
    """The model could not be loaded, with a reason worth reading."""

@dataclass
class TestResults:
    """Store test results from the CAG/Non-CAG process."""
    cache_time: List[float]
    generate_time: List[float]
    similarity: List[float]
    prompts: List[str]
    responses: List[str]
    ground_truths: List[str]
    timestamps: List[str]
    prepare_time: float = 0.0

    @property
    def avg_similarity(self) -> float:
        return sum(self.similarity) / len(self.similarity)

    @property
    def avg_cache_time(self) -> float:
        return sum(self.cache_time) / len(self.cache_time)

    @property
    def avg_generate_time(self) -> float:
        return sum(self.generate_time) / len(self.generate_time)

class CAGModel:
    """Core CAG model logic."""
    
    def __init__(self, hf_token: str):
        self.hf_token = hf_token
        self.model = None
        self.tokenizer = None
        self.bert_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True
        )

    def load_model(self, model_name: str, quantized: bool = False) -> bool:
        """Load the model with optional quantization.

        Raises :class:`ModelLoadError` rather than returning False. The
        original printed the exception and returned False, so the caller got a
        bare "it did not work" with no way to tell a gated repo from a missing
        token from an out-of-memory error - three problems with three
        completely different fixes.
        """
        try:
            if quantized:
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    quantization_config=self.quantization_config,
                    device_map="auto",
                    trust_remote_code=True,
                    token=self.hf_token or None
                )
            else:
                torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    device_map="auto",
                    torch_dtype=torch_dtype,
                    low_cpu_mem_usage=True,
                    trust_remote_code=True,
                    token=self.hf_token or None
                )
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, token=self.hf_token or None)
            return True
        except Exception as exc:
            message = str(exc)
            if "401" in message or "gated" in message.lower() or "restricted" in message.lower():
                raise ModelLoadError(
                    f"{model_name} is a gated repository. Set HF_TOKEN and "
                    f"accept the licence at https://huggingface.co/{model_name} "
                    f"with the same account. Or pick one of the ungated models "
                    f"- they need no token at all."
                ) from exc
            if "out of memory" in message.lower() or "CUDA" in message:
                raise ModelLoadError(
                    f"Not enough memory to load {model_name}: {exc}. "
                    f"Try a smaller model, or enable quantization."
                ) from exc
            raise ModelLoadError(f"Could not load {model_name}: {exc}") from exc

    def generate_response(self, prompt: str, past_key_values: DynamicCache = None, max_tokens: int = 300) -> str:
        """Generate model response."""
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.model.device)
        output_ids = input_ids.clone()
        next_token = input_ids

        with torch.no_grad():
            for _ in range(max_tokens):
                outputs = self.model(
                    input_ids=next_token,
                    past_key_values=past_key_values,
                    use_cache=True
                )
                next_token = outputs.logits[:, -1, :].argmax(dim=-1).unsqueeze(-1).to(self.model.device)
                past_key_values = outputs.past_key_values
                output_ids = torch.cat([output_ids, next_token], dim=1)

                eos_token_ids = self.model.config.eos_token_id
                if isinstance(eos_token_ids, int):
                    eos_token_ids = [eos_token_ids]
                if next_token.item() in eos_token_ids:
                    break

        output = output_ids[:, input_ids.shape[-1]:]
        return self.tokenizer.decode(output[0], skip_special_tokens=True)

    def prepare_cache(self, documents: str, instruction: str = None) -> Tuple[DynamicCache, float]:
        """Prepare KV cache."""
        start_time = time()
        instruction = instruction or "Answer the question with a short answer."
        prompt = self._create_prompt(documents, instruction)

        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.model.device)
        past_key_values = DynamicCache()

        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                past_key_values=past_key_values,
                use_cache=True,
                output_attentions=False,
                output_hidden_states=False
            )

        return outputs.past_key_values, time() - start_time

    def process_questions(self, dataset: List[Tuple[str, str]], documents: str, use_cache: bool = True, 
                         progress_callback=None) -> TestResults:
        """Process a batch of questions."""
        results = TestResults([], [], [], [], [], [], [])

        # Build the KV cache over the whole document exactly once. Every
        # question then reuses a copy of it, which is the entire idea.
        shared_cache = None
        origin_len = 0
        if use_cache:
            shared_cache, build_time = self.prepare_cache(documents)
            origin_len = self._cache_length(shared_cache)
            results.prepare_time = build_time

        for idx, (question, ground_truth) in enumerate(dataset):
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            if progress_callback:
                progress_callback((idx + 1) / len(dataset))

            prompt = self._create_question_prompt(question, documents if not use_cache else "")

            if use_cache:
                # Reuse the cache built once, above. The original loaded it
                # from ./data_cache/cache_knowledges.pt on every question - a
                # file nothing in this project ever wrote, so the cached path
                # could only ever raise FileNotFoundError. Keeping it in
                # memory is also the point: cache-augmented generation is
                # about not re-reading the document, not about disk.
                cache_t1 = time()
                cache = copy.deepcopy(shared_cache)
                self._clean_cache(cache, origin_len)
                cache_time = time() - cache_t1
            else:
                cache_time = 0
                cache = None

            start_time = time()
            response = self.generate_response(prompt, cache)
            gen_time = time() - start_time
            similarity = self._calculate_similarity(response, ground_truth)

            self._store_result(results, question, response, ground_truth, similarity, cache_time, gen_time)

        return results

    def _create_prompt(self, documents: str, instruction: str) -> str:
        return f"""
        <|begin_of_text|>
        <|start_header_id|>system<|end_header_id|>
        You are an assistant for giving short answers based on given context.
        <|eot_id|>
        <|start_header_id|>user<|end_header_id|>
        Context information is below.
        ------------------------------------------------
        {documents}
        ------------------------------------------------
        {instruction}
        Question:
        """

    def _create_question_prompt(self, question: str, context: str = "") -> str:
        if context:
            return self._create_prompt(context, "") + f"{question}<|eot_id|>\n<|start_header_id|>assistant<|end_header_id|>"
        return f"{question}<|eot_id|>\n<|start_header_id|>assistant<|end_header_id|>"

    def _calculate_similarity(self, response: str, ground_truth: str) -> float:
        response_embedding = self.bert_model.encode(response, convert_to_tensor=True)
        truth_embedding = self.bert_model.encode(ground_truth, convert_to_tensor=True)
        return torch.cosine_similarity(response_embedding, truth_embedding, dim=0).item()

    @staticmethod
    def _cache_length(kv: DynamicCache) -> int:
        """How many tokens the cache currently holds.

        transformers 5.x replaced `DynamicCache.key_cache` with `.layers`, so
        reaching into `key_cache[0].shape[-2]` raises AttributeError there.
        `get_seq_length()` is the supported API and works on both.
        """
        try:
            return int(kv.get_seq_length())
        except Exception:
            return kv.key_cache[0].shape[-2]

    def _clean_cache(self, kv: DynamicCache, origin_len: int):
        """Trim the cache back to the document, dropping the last answer.

        Without this the previous question and its answer stay in the cache
        and leak into the next one. `crop` is the supported API in
        transformers 5.x; the manual slice is the pre-5.x fallback.
        """
        try:
            current = int(kv.get_seq_length())
            extra = current - origin_len
            if extra <= 0:
                return
            # transformers 5.16 deprecated crop(positive_length) in favour of
            # crop(-n), meaning "remove n tokens". Passing the negative form
            # keeps this working past 5.18, where the positive form is removed.
            kv.crop(-extra)
            return
        except (AttributeError, TypeError):
            pass
        for i in range(len(kv.key_cache)):
            kv.key_cache[i] = kv.key_cache[i][:, :, :origin_len, :]
            kv.value_cache[i] = kv.value_cache[i][:, :, :origin_len, :]

    def _store_result(self, results: TestResults, question: str, response: str,
                     ground_truth: str, similarity: float, cache_time: float, gen_time: float):
        results.prompts.append(question)
        results.responses.append(response)
        results.ground_truths.append(ground_truth)
        results.similarity.append(similarity)
        results.cache_time.append(cache_time)
        results.generate_time.append(gen_time)
        results.timestamps.append(datetime.now().strftime("%H:%M:%S"))