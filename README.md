# Talk2Table
A webapp to talk to any table or databases (SQL or Excel) in plain English

# 🔍 Sherlock Protocol

*A privacy‑first execution strategy for Talk2Table*

---

## 1. What problem does it solve?

When you let a large‑language model (LLM) inspect raw tables it may:

* **Leak sensitive rows** via its responses.
* **Choke on large datasets** (token limits).
* **Waste tokens** serialising data it does not need.

Sherlock flips the script: **the model never sees a single row.** Instead, it solves each request “from the couch”, sending controlled instructions to a sandbox that owns the dataframe.

```
┌──────────────┐   ① natural‑language query   ┌─────────────┐
│   **User**   │ ───────────────────────────▶ │    LLM      │  (blind)
└──────────────┘                              └─────────────┘
                                                  │ ② helper calls only
                                                  ▼
                                            ┌────────────────┐
                                            │ Sherlock Sandbox│
                                            │   (Python)     │
                                            ├────────────────┤
                                            │ Dataframe copy │
                                            └────────────────┘
                                                  │ ③ safe result (counts, aggregates…)
                                                  ▼
                                           back to the user ✨
```

## 2. How does it work?

1. **Whitelist of primitives** – `equal_filter`, `numeric_filter`, `safe_agg` & friends are the *only* gateway to the dataframe.
2. **Blind execution** – the LLM composes a plan using those helpers; the plan runs in an isolated Python env.
3. **No raw export** – each primitive returns either a *copy* of the subset (still inside the sandbox) or aggregated scalars.  The final answer shown to the user is a small, safe object (counts, stats, plots).
4. **Retry‑and‑refine loop** – if code fails validation, the LLM self‑corrects without ever requesting raw data.

## 3. Key helper categories

| Category          | Examples                                                           |
| ----------------- | ------------------------------------------------------------------ |
| **Cleaning**      | `standardise_df`, `standardise_column`                             |
| **Fuzzy search**  | `fuzzy_filter`, `best_match`, `list_distinct`                      |
| **Numeric / Agg** | `numeric_range`, `safe_agg`                                        |
| **Row filters**   | `equal_filter`, `prefix_filter`, `numeric_filter`, `null_filter` … |
| **Combinators**   | `row_any`                                                          |

Full API lives in [`sherlock_protocol.py`](./sherlock_protocol.py).

## 4. Pros & limitations

| Aspect             | **Sherlock Protocol (blind)**                           | **Direct LLM access to data**                     |
| ------------------ | ------------------------------------------------------- | ------------------------------------------------- |
| **Privacy**        | ✅ Raw rows never leave runtime                          | ❌ High risk of leakage                            |
| **GDPR / PII**     | ✅ Derived values only → easier compliance               | ❌ Needs heavy redaction & review                  |
| **Token usage**    | ✅ Minimal – model sees helper calls only                | ❌ Token budget explodes on large tables           |
| **Performance**    | ✅ Vectorised Pandas in‑sandbox; no upload latency       | ⚠️ Depends on context size & transfer time        |
| **Flexibility**    | ⚠️ Must express query via helpers (covers \~99 % needs) | ✅ Arbitrary SQL / pandas possible                 |
| **Debuggability**  | ✅ Deterministic helper set, easy to unit‑test           | ❌ Harder to trace arbitrary code the model writes |
| **Learning curve** | ⚠️ Need to learn helper vocabulary                      | ✅ Natural language only                           |

## 5. When might you *not* use Sherlock?

* Exploratory data‑science sessions where raw inspection is *required*.
* Tiny, completely non‑sensitive datasets where privacy is not a concern.

For 90 % of BI & report queries, Sherlock hits the sweet spot of **speed, safety, and transparency**.

---

> *"You know my methods.  Apply them."* — **Sherlock Holmes to Dr. Watson**
>
> Sherlock Protocol gives the LLM disciplined methods while your raw evidence stays locked safely on‑prem 🔒

