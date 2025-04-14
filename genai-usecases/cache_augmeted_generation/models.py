from dataclasses import dataclass
from typing import List, Tuple
from time import time
from datetime import datetime
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
        """Load the model with optional quantization."""
        try:
            if quantized:
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    quantization_config=self.quantization_config,
                    device_map="auto",
                    trust_remote_code=True,
                    token=self.hf_token
                )
            else:
                torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    device_map="auto",
                    torch_dtype=torch_dtype,
                    low_cpu_mem_usage=True,
                    trust_remote_code=True,
                    token=self.hf_token
                )
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, token=self.hf_token)
            return True
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            return False

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

                if next_token.item() in self.model.config.eos_token_id:
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

        for idx, (question, ground_truth) in enumerate(dataset):
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            if progress_callback:
                progress_callback((idx + 1) / len(dataset))

            prompt = self._create_question_prompt(question, documents if not use_cache else "")

            if use_cache:
                cache_t1 = time()
                cache = torch.load("./data_cache/cache_knowledges.pt", weights_only=True)
                cache_time = time() - cache_t1
                self._clean_cache(cache, cache.key_cache[0].shape[-2])
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

    def _clean_cache(self, kv: DynamicCache, origin_len: int):
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