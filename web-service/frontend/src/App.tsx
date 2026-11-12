import { useEffect, useState } from 'react'
import type { FormEvent } from 'react'
import { API_URL } from './config'

interface Metadata {
  categories: Record<string, string[]>
  fx_margin_by_destination: Record<string, number>
  fx_margin_overall: number
  year_max: number
}

interface Prediction {
  predicted_cost_pct: number
  estimated_fee_usd: number | null
}

const FIELDS: { key: string; field: string; label: string }[] = [
  { key: 'destination_name', field: 'destination_name', label: 'Destination country' },
  { key: 'firm_type_clean', field: 'firm_type_clean', label: 'Provider type' },
  { key: 'payment_instrument_group', field: 'payment_instrument_group', label: 'Payment method' },
  { key: 'access_point', field: 'access point', label: 'Where you send from' },
  { key: 'receiving_network_coverage', field: 'receiving network coverage', label: 'Receiving network coverage' },
  { key: 'speed_actual', field: 'speed actual', label: 'Delivery speed' },
  { key: 'pickup_method', field: 'pickup method', label: 'Pickup method' },
]

function App() {
  const [meta, setMeta] = useState<Metadata | null>(null)
  const [values, setValues] = useState<Record<string, string>>({})
  const [fxMargin, setFxMargin] = useState('')
  const [year, setYear] = useState('')
  const [nonTransparent, setNonTransparent] = useState(false)
  const [amount, setAmount] = useState('')
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [result, setResult] = useState<Prediction | null>(null)

  useEffect(() => {
    fetch(`${API_URL}/metadata`)
      .then((r) => {
        if (!r.ok) throw new Error(`Backend returned ${r.status}`)
        return r.json() as Promise<Metadata>
      })
      .then((m) => {
        setMeta(m)
        const initial: Record<string, string> = {}
        for (const f of FIELDS) initial[f.key] = m.categories[f.field][0]
        setValues(initial)
        setFxMargin(String(m.fx_margin_by_destination[initial.destination_name] ?? m.fx_margin_overall))
        setYear(String(m.year_max))
      })
      .catch((e: Error) => setError(`Could not reach the backend at ${API_URL}: ${e.message}`))
  }, [])

  const onChange = (key: string, value: string) => {
    setValues((v) => ({ ...v, [key]: value }))
    if (key === 'destination_name' && meta) {
      setFxMargin(String(meta.fx_margin_by_destination[value] ?? meta.fx_margin_overall))
    }
  }

  const onSubmit = async (e: FormEvent) => {
    e.preventDefault()
    setLoading(true)
    setError(null)
    setResult(null)
    try {
      const res = await fetch(`${API_URL}/predict`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          ...values,
          cc1_fx_margin: Number(fxMargin),
          year: Number(year),
          is_non_transparent: nonTransparent,
          transfer_amount: amount === '' ? null : Number(amount),
        }),
      })
      if (!res.ok) {
        const body = await res.json().catch(() => null)
        const detail = body?.detail
        throw new Error(
          typeof detail === 'string' ? detail : detail ? JSON.stringify(detail) : `Status ${res.status}`,
        )
      }
      setResult((await res.json()) as Prediction)
    } catch (err) {
      setError(`Prediction failed: ${(err as Error).message}`)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="page">
      <h1>Remittance cost estimator</h1>
      <p className="subtitle">
        Predict the total cost of sending money (as % of the amount) from the U.S.
      </p>
      {meta && (
        <form onSubmit={onSubmit}>
          {FIELDS.map((f) => (
            <label key={f.key}>
              {f.label}
              <select value={values[f.key]} onChange={(e) => onChange(f.key, e.target.value)}>
                {meta.categories[f.field].map((o) => (
                  <option key={o} value={o}>
                    {o}
                  </option>
                ))}
              </select>
            </label>
          ))}
          <label>
            Exchange-rate margin (%)
            <input type="number" step="any" required value={fxMargin} onChange={(e) => setFxMargin(e.target.value)} />
            <span className="hint">Pre-filled with the historical average for this destination.</span>
          </label>
          <label>
            Year
            <input type="number" required min={2000} max={2100} value={year} onChange={(e) => setYear(e.target.value)} />
          </label>
          <label className="checkbox">
            <input type="checkbox" checked={nonTransparent} onChange={(e) => setNonTransparent(e.target.checked)} />
            Provider does not disclose its fees transparently
          </label>
          <label>
            Transfer amount (USD, optional)
            <input type="number" min={0} step="any" value={amount} onChange={(e) => setAmount(e.target.value)} />
          </label>
          <button type="submit" disabled={loading}>
            {loading ? 'Predicting…' : 'Predict cost'}
          </button>
        </form>
      )}
      {error && <div className="error">{error}</div>}
      {result && (
        <div className="result">
          <div>Predicted total cost</div>
          <div className="big">{result.predicted_cost_pct.toFixed(2)}%</div>
          {result.estimated_fee_usd !== null && (
            <div>Estimated fee: ${result.estimated_fee_usd.toFixed(2)}</div>
          )}
        </div>
      )}
    </div>
  )
}

export default App
