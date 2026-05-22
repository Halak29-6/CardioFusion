/**
 * CardioFusion API Client
 * Centralises all HTTP calls to the gateway service.
 */

const GATEWAY = process.env.REACT_APP_GATEWAY_URL || "";

/**
 * Generate a synthetic 1000-point ECG waveform for demo purposes.
 * In production, replace with actual device upload / waveform parsing.
 */
export function generateDemoECG(seed = 42) {
  const waveform = [];
  for (let i = 0; i < 1000; i++) {
    const t    = (i / 1000) * 4 * Math.PI;
    const hr   = 75;
    const val  =
      Math.sin(t * hr / 60) +
      0.3 * Math.sin(2 * t * hr / 60) +
      0.05 * (Math.random() - 0.5);
    waveform.push(parseFloat(val.toFixed(5)));
  }
  return waveform;
}

/**
 * Submit EHR + ECG data for analysis.
 * Returns the full enriched JSON handshake from the gateway.
 */
export async function analysePatient(ecg_waveform, ehr) {
  const response = await fetch(`${GATEWAY}/analyse`, {
    method:  "POST",
    headers: { "Content-Type": "application/json" },
    body:    JSON.stringify({ ecg_waveform, ehr }),
  });

  if (!response.ok) {
    const err = await response.json().catch(() => ({}));
    throw new Error(err.error || `Gateway error ${response.status}`);
  }

  return response.json();
}
