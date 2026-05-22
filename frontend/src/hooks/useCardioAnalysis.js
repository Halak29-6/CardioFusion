import { useState, useCallback } from "react";
import { analysePatient, generateDemoECG } from "../utils/api";

const INITIAL_EHR = {
  age:               65,
  sex:               1,
  height:            172,
  weight:            85,
  ejection_fraction: 35,
  serum_creatinine:  1.6,
  sodium:            134,
  bp_systolic:       105,
  bp_diastolic:      70,
  smoking_status:    1,
  bmi:               28.7,
};

export function useCardioAnalysis() {
  const [ehr,         setEhr]         = useState(INITIAL_EHR);
  const [ecgData,     setEcgData]     = useState(() => generateDemoECG());
  const [result,      setResult]      = useState(null);
  const [loading,     setLoading]     = useState(false);
  const [shapLoading, setShapLoading] = useState(false);
  const [error,       setError]       = useState(null);

  const updateEhr = useCallback((field, value) => {
    setEhr((prev) => ({ ...prev, [field]: Number(value) }));
  }, []);

  const regenerateECG = useCallback(() => {
    setEcgData(generateDemoECG(Math.random()));
  }, []);

  const runAnalysis = useCallback(async () => {
    setLoading(true);
    setShapLoading(true);
    setError(null);
    setResult(null);

    try {
      const data = await analysePatient(ecgData, ehr);
      setResult(data);

      // SHAP may still be computing – reflect that in UI
      if (!data.shap_ready) {
        setShapLoading(true);
        // In a real app you'd poll /explain/{job_id} here
        // For demo: simulate resolution after 3s
        setTimeout(() => setShapLoading(false), 3000);
      } else {
        setShapLoading(false);
      }
    } catch (err) {
      setError(err.message || "Analysis failed. Please try again.");
    } finally {
      setLoading(false);
    }
  }, [ecgData, ehr]);

  return {
    ehr, updateEhr,
    ecgData, regenerateECG,
    result, loading, shapLoading, error,
    runAnalysis,
  };
}
