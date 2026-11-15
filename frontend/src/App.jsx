import { useState } from "react";

const MODELS = ["popularity", "content", "collaborative", "hybrid"];
const MISSING = ["popularity", "global_mean", "drop"];
const METRICS = ["precision", "recall", "ndcg", "coverage", "missing_rate"];

const start = [
  { name: "Content (k=30)", type: "content", params: '{"k": 30}', missing: "popularity" },
  { name: "Collaborative (k=30)", type: "collaborative", params: '{"k": 30}', missing: "popularity" },
  { name: "Hybrid adaptive", type: "hybrid", params: '{"alpha": "adaptive"}', missing: "popularity" },
  { name: "Hybrid alpha 0.5", type: "hybrid", params: '{"alpha": 0.5}', missing: "popularity" },
];

async function api(path, options) {
  const res = await fetch(`/api${path}`, options);
  const body = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(body.detail ? JSON.stringify(body.detail) : `Request failed (${res.status})`);
  return body;
}

export default function App() {
  const [configs, setConfigs] = useState(start);
  const [k, setK] = useState(10);
  const [results, setResults] = useState([]);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [user, setUser] = useState(0);
  const [recModel, setRecModel] = useState("hybrid");
  const [recs, setRecs] = useState(null);

  const edit = (i, field, value) => setConfigs(configs.map((c, j) => (j === i ? { ...c, [field]: value } : c)));

  async function run() {
    setBusy(true); setError("");
    try {
      const payload = configs.map((c) => ({ ...c, params: JSON.parse(c.params || "{}") }));
      const data = await api("/evaluate", { method: "POST", headers: { "Content-Type": "application/json" },
                                            body: JSON.stringify({ configs: payload, k: Number(k) }) });
      setResults(data.results);
    } catch (e) {
      setError(e instanceof SyntaxError ? "Params must be valid JSON, e.g. {\"k\": 20}" : e.message);
    } finally { setBusy(false); }
  }

  async function showRecs() {
    setError("");
    try { setRecs(await api(`/recommend/${user}?model=${recModel}&k=${k}`)); }
    catch (e) { setError(e.message); }
  }

  const best = (m) => Math.max(...results.map((r) => r[m]));

  return (
    <main>
      <h1>Recommender Lab</h1>
      <p className="lede">Set up model configurations, run them on the same held-out ratings, and compare ranking quality side by side.</p>

      <h2>Configurations</h2>
      <div className="panel">
        {configs.map((c, i) => (
          <div className="row" key={i}>
            <input aria-label="Name" value={c.name} onChange={(e) => edit(i, "name", e.target.value)} />
            <select aria-label="Model" value={c.type} onChange={(e) => edit(i, "type", e.target.value)}>
              {MODELS.map((m) => <option key={m}>{m}</option>)}
            </select>
            <input aria-label="Params (JSON)" value={c.params} onChange={(e) => edit(i, "params", e.target.value)} />
            <select aria-label="Missing-prediction strategy" value={c.missing} onChange={(e) => edit(i, "missing", e.target.value)}>
              {MISSING.map((m) => <option key={m}>{m}</option>)}
            </select>
            <button onClick={() => setConfigs(configs.filter((_, j) => j !== i))} aria-label="Remove configuration">Remove</button>
          </div>
        ))}
        <div className="bar">
          <button onClick={() => setConfigs([...configs, { name: `Config ${configs.length + 1}`, type: "content", params: "{}", missing: "popularity" }])}>Add configuration</button>
          <label>K <input type="number" min="1" max="50" value={k} onChange={(e) => setK(e.target.value)} /></label>
          <button className="primary" disabled={busy || !configs.length} onClick={run}>{busy ? "Evaluating…" : "Run evaluation"}</button>
        </div>
        <p className="hint">Missing strategy decides what happens when a model cannot score an item: fall back to item popularity, the global mean, or leave the item out of the ranking.</p>
        {error && <p className="err" role="alert">{error}</p>}
      </div>

      {results.length > 0 && (
        <>
          <h2>Results at K = {results[0].k}</h2>
          <div className="panel scroll">
            <table>
              <thead><tr><th>Configuration</th>{METRICS.map((m) => <th key={m}>{m === "ndcg" ? "NDCG@K" : m === "precision" ? "Precision@K" : m === "recall" ? "Recall@K" : m === "coverage" ? "Coverage" : "Missing rate"}</th>)}<th>Seconds</th></tr></thead>
              <tbody>
                {results.map((r) => (
                  <tr key={r.name}>
                    <td>{r.name}</td>
                    {METRICS.map((m) => (
                      <td key={m} className={m !== "missing_rate" && r[m] === best(m) ? "best" : ""}>{r[m].toFixed(3)}</td>
                    ))}
                    <td>{r.seconds}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <p className="hint">Highlighted cells are the best value in each ranking column. Missing rate is the share of unrated items the model could not score before fallback.</p>
        </>
      )}

      <h2>Recommendations for a user</h2>
      <div className="panel">
        <div className="bar" style={{ marginTop: 0 }}>
          <label>User <input type="number" min="0" value={user} onChange={(e) => setUser(e.target.value)} /></label>
          <select aria-label="Model" value={recModel} onChange={(e) => setRecModel(e.target.value)} style={{ width: "auto" }}>
            {MODELS.map((m) => <option key={m}>{m}</option>)}
          </select>
          <button className="primary" onClick={showRecs}>Show top {k}</button>
        </div>
        {recs && (
          <div className="scroll">
            <p className="hint">User {recs.user_idx} has {recs.ratings_in_train} training ratings{recs.alpha != null && <> · collaborative weight α = {recs.alpha.toFixed(2)}</>}</p>
            <table>
              <thead><tr><th>Title</th><th className="l">Genres</th><th>Score</th></tr></thead>
              <tbody>{recs.recommendations.map((r) => <tr key={r.item_idx}><td>{r.title}</td><td className="l">{r.genres}</td><td>{r.score}</td></tr>)}</tbody>
            </table>
          </div>
        )}
      </div>
    </main>
  );
}
