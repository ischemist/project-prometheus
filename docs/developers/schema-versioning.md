---
icon: lucide/git-branch
---

# Schema Versioning

CalcFlow serializes `CalculationInput` and `CalculationResult` to JSON. Schema versioning ensures that JSON produced by an older version of CalcFlow can still be loaded by a newer version.

## Two Version Fields

Every serialized object includes two version fields:

```json
{
  "calcflow_version": "0.5.0",
  "schema_version": 2,
  ...
}
```

| Field | Type | Purpose |
| :--- | :--- | :--- |
| `calcflow_version` | string (semver) | Provenance — which version of the library produced this dump. Never used in loading logic. |
| `schema_version` | integer | Structural compatibility — drives migration logic on load. |

**These are independent.** A patch release that fixes a parser bug does not touch `schema_version`. A minor release that renames a field does.

## Where the Constants Live

```python
# calcflow/common/results.py
RESULT_SCHEMA_VERSION: int = 2

# calcflow/common/input.py
INPUT_SCHEMA_VERSION: int = 1
```

Each model tracks its own version independently. A change to `CalculationResult`'s schema does not require bumping `INPUT_SCHEMA_VERSION`.

## When to Bump `schema_version`

**Bump** when an old dump would fail to load correctly with the new code:

- Renaming a field
- Removing a field
- Restructuring a nested object (e.g. changing a flat dict to a nested object)
- Changing a field's type (e.g. `float` to `str`)
- Adding a **required** field (one with no default)

**Do not bump** when old dumps remain loadable without changes:

- Adding an optional field with a default value
- Fixing a parser bug (the schema structure is unchanged)
- Changing a default value
- Renaming an internal variable that doesn't appear in serialized output

!!! tip "The rule of thumb"
    If `from_dict(old_dump)` would raise an error or silently produce wrong data, bump `schema_version`. If it would work correctly, don't.

## Writing a Migration

Migrations live in the `_migrate()` class method of each model. They are sequential: v1→v2, then v2→v3, and so on. Never skip a step.

```python
# calcflow/common/results.py

RESULT_SCHEMA_VERSION = 3

@classmethod
def _migrate(cls, data: dict) -> dict:
    version = data.get("schema_version", 1)

    if version < 2:
        # v1 -> v2: geometry fields changed from bare atom lists to Geometry objects
        for key in ("input_geometry", "final_geometry"):
            if isinstance(data.get(key), list):
                data[key] = {"comment": "", "atoms": data[key]}
        version = 2

    if version < 3:
        # v2 -> v3: renamed "atomic_charges" list items from "method_name" to "method"
        for charge_dict in data.get("atomic_charges", []):
            if "method_name" in charge_dict:
                charge_dict["method"] = charge_dict.pop("method_name")
        version = 3

    data["schema_version"] = version
    return data
```

The migration is called by `from_dict()` before deserializing:

```python
@classmethod
def from_dict(cls, data: dict) -> "CalculationResult":
    data = cls._migrate(dict(data))  # shallow copy before mutating
    # ... deserialize fields ...
```

## Detecting Forward Incompatibility

If a dump has a `schema_version` higher than the current `RESULT_SCHEMA_VERSION`, the code cannot safely load it — it was produced by a newer version of CalcFlow that added fields this version doesn't know about.

```python
stored_version = data.get("schema_version", 1)
if stored_version > RESULT_SCHEMA_VERSION:
    raise ConfigurationError(
        f"Cannot load result: schema version {stored_version} is newer than "
        f"this version of CalcFlow supports ({RESULT_SCHEMA_VERSION}). "
        f"Please upgrade calcflow."
    )
```

## Example: Adding an Optional Field (No Bump Needed)

Adding `timing: TimingResults | None = None` to `CalculationResult`:

1. Add the field with a default of `None`.
2. Old dumps that don't include `timing` will deserialize with `timing=None` — correct behavior.
3. No migration step needed.
4. **Do not bump `schema_version`.**

## Example: Renaming a Field (Bump Required)

Renaming `CalculationResult.scf_results` to `CalculationResult.scf`:

1. Bump `RESULT_SCHEMA_VERSION` from 2 to 3.
2. Add a migration step in `_migrate()`:

    ```python
    if version < 3:
        if "scf_results" in data:
            data["scf"] = data.pop("scf_results")
        version = 3
    ```

3. Update `to_dict()` to use the new key `"scf"`.
4. Old dumps with `"scf_results"` will be transparently migrated.
