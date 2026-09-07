"""The app registry. No API key, no network."""

import inspect

import pytest

from services import registry


def test_every_app_is_registered_under_its_own_id():
    for app_id, app in registry.REGISTRY.items():
        assert app.id == app_id


def test_every_app_declares_at_least_one_field():
    for app in registry.REGISTRY.values():
        assert app.fields, f"{app.id} declares no inputs"


def test_every_runner_is_async():
    # The underlying workflows are coroutines; a sync runner would return a
    # coroutine object to the API and serialise as null.
    for app in registry.REGISTRY.values():
        assert inspect.iscoroutinefunction(app.runner), f"{app.id} runner is sync"


def test_select_fields_default_to_one_of_their_options():
    for app in registry.REGISTRY.values():
        for field in app.fields:
            if field.kind == "select":
                assert field.default in field.options, \
                    f"{app.id}.{field.name} default is not in options"


def test_catalogue_is_json_serialisable():
    import json
    json.dumps(registry.catalogue())


def test_catalogue_matches_the_registry():
    assert {a["id"] for a in registry.catalogue()} == set(registry.REGISTRY)


def test_get_raises_for_unknown_app():
    with pytest.raises(KeyError):
        registry.get("does-not-exist")


class TestEventDrivenSetup:
    """Reactive agents must exist before the workflow publishes events.

    `start_workflow` publishes onto an event bus. Agents subscribe when they
    are constructed, so with none registered the call succeeds, returns None,
    and nothing at all happens - a silent no-op that looks like a working run.
    """

    def test_event_driven_exposes_an_agent_count_input(self):
        app = registry.get("event_driven")
        names = {f.name for f in app.fields}
        assert "agent_count" in names

    def test_agent_count_defaults_to_at_least_one(self):
        app = registry.get("event_driven")
        field = next(f for f in app.fields if f.name == "agent_count")
        assert int(field.default) >= 1


class TestPromptChainingDefaults:
    def test_default_steps_carry_no_template_placeholders(self):
        """Step prompts are plain instructions.

        The chain supplies the topic and the previous output itself, so a
        literal "{topic}" in a step collides with its template variables and
        the run fails with "Input to ChatPromptTemplate is missing variables".
        """
        import inspect

        source = inspect.getsource(registry.run_prompt_chaining)
        start = source.index("chain_steps = steps or")
        end = source.index("return", start)
        defaults = source[start:end]
        for placeholder in ("{topic}", "{previous_output}", "{input}"):
            assert placeholder not in defaults, (
                f"default step prompt contains {placeholder}"
            )


class TestCrewConfigs:
    """The CrewAI crews load their agents, tasks and crew from YAML.

    Those files lived beside each crew's old script. After the move they would
    all have resolved to one shared `services/apps/config`, colliding with
    each other - so each crew now has its own subdirectory, and this checks
    the files are actually there.
    """

    CREWS = ["code_review_crew", "content_creation_crew",
             "data_analysis_crew", "research_assistant_crew"]

    def test_every_crew_has_its_own_config_directory(self):
        from pathlib import Path
        root = Path(__file__).resolve().parent.parent / "services" / "apps" / "config"
        for crew in self.CREWS:
            for name in ("agents.yaml", "tasks.yaml", "crew.yaml"):
                path = root / crew / name
                assert path.exists(), f"missing {crew}/{name}"
                assert path.read_text(encoding="utf-8").strip(), f"{crew}/{name} is empty"

    def test_config_yaml_parses(self):
        import yaml
        from pathlib import Path
        root = Path(__file__).resolve().parent.parent / "services" / "apps" / "config"
        for crew in self.CREWS:
            for name in ("agents.yaml", "tasks.yaml", "crew.yaml"):
                yaml.safe_load((root / crew / name).read_text(encoding="utf-8"))
