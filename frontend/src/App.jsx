import { useState } from "react";
import "./App.css";

function App() {
  const validateHealth = (health) => {
  const errors = {};

  if (health.age < 18 || health.age > 100) {
    errors.age = "Age must be between 18 and 100.";
  }

  if (health.height_cm < 50 || health.height_cm > 250) {
    errors.height_cm = "Height must be between 50 and 250 cm.";
  }

  if (health.weight_kg < 20 || health.weight_kg > 300) {
    errors.weight_kg = "Weight must be between 20 and 300 kg.";
  }

  if (health.ap_hi < 70 || health.ap_hi > 250) {
    errors.ap_hi = "Systolic BP must be between 70 and 250.";
  }

  if (health.ap_lo < 40 || health.ap_lo > 150) {
    errors.ap_lo = "Diastolic BP must be between 40 and 150.";
  }

  if (health.HbA1c_level < 3 || health.HbA1c_level > 20) {
    errors.HbA1c_level = "HbA1c must be between 3 and 20.";
  }

  if (
    health.blood_glucose_level < 40 ||
    health.blood_glucose_level > 600
  ) {
    errors.blood_glucose_level =
      "Blood glucose must be between 40 and 600.";
  }

  return errors;
};

const validateFinance = (finance) => {
  const errors = {};

  if (finance.income <= 0) {
    errors.income = "Monthly income must be greater than 0.";
  }

  if (finance.fixed_expenses < 0) {
    errors.fixed_expenses = "Fixed expenses cannot be negative.";
  }

  if (finance.variable_expenses < 0) {
    errors.variable_expenses = "Variable expenses cannot be negative.";
  }

  if (finance.emis < 0) {
    errors.emis = "EMIs cannot be negative.";
  }

  if (finance.savings_balance < 0) {
    errors.savings_balance = "Savings balance cannot be negative.";
  }

  if (
    finance.fixed_expenses +
      finance.variable_expenses +
      finance.emis >
    finance.income
  ) {
    errors.expenses =
      "Total monthly expenses cannot exceed monthly income.";
  }

  return errors;
};
const validateInsurance = (insurance) => {
  const errors = {};

  if (insurance.sum_insured <= 0) {
    errors.sum_insured = "Sum insured must be greater than 0.";
  }

  if (insurance.annual_premium < 0) {
    errors.annual_premium = "Annual premium cannot be negative.";
  }

  if (insurance.annual_income <= 0) {
    errors.annual_income = "Annual income must be greater than 0.";
  }

  if (insurance.annual_premium > insurance.annual_income) {
    errors.premium =
      "Annual premium cannot be greater than annual income.";
  }

  return errors;
};


  const [query, setQuery] = useState("");
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(false);

  const [profile, setProfile] = useState({
    health: {
      age: 30,
      gender: "male",
      height_cm: 175,
      weight_kg: 70,
      ap_hi: 120,
      ap_lo: 80,
      cholesterol: 1,
      gluc: 1,
      smoke: 0,
      alco: 0,
      active: 1,
      hypertension: 0,
      heart_disease: 0,
      smoking_history: "never",
      HbA1c_level: 5.2,
      blood_glucose_level: 90,
    },
    finance: {
      income: 60000,
      fixed_expenses: 15000,
      variable_expenses: 10000,
      emis: 5000,
      savings_balance: 100000,
    },
    insurance: {
      sum_insured: 500000,
      annual_premium: 20000,
      annual_income: 720000,
      existing_riders: [],
    },
  });

  const updateHealth = (field, value) => {
    setProfile((prev) => ({
      ...prev,
      health: {
        ...prev.health,
        [field]: value,
      },
    }));
  };

  const updateFinance = (field, value) => {
    setProfile((prev) => ({
      ...prev,
      finance: {
        ...prev.finance,
        [field]: value,
      },
    }));
  };

  const updateInsurance = (field, value) => {
    setProfile((prev) => ({
      ...prev,
      insurance: {
        ...prev.insurance,
        [field]: value,
      },
    }));
  };

  // const askTwinLife = async () => {
  //   if (!query.trim()) return;
  const askTwinLife = async () => {
  if (!query.trim()) return;

  const healthErrors = validateHealth(profile.health);

  if (Object.keys(healthErrors).length > 0) {
    alert(Object.values(healthErrors).join("\n"));
    return;
  }
  const financeErrors = validateFinance(profile.finance);

if (Object.keys(financeErrors).length > 0) {
  alert(Object.values(financeErrors).join("\n"));
  return;
}
const insuranceErrors = validateInsurance(profile.insurance);

if (Object.keys(insuranceErrors).length > 0) {
  alert(Object.values(insuranceErrors).join("\n"));
  return;
}

    setLoading(true);
    setData(null);

    try {
      const res = await fetch("http://127.0.0.1:8000/query", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          query,
          profile,
        }),
      });

      const result = await res.json();
      console.log("RAG DATA:", result.rag);
      setData(result);
    } catch (error) {
      setData({
        error: "Could not connect to TwinLife AI backend.",
      });
    }

    setLoading(false);
  };

  const health = data?.health;
  const finance = data?.finance;
  const insurance = data?.insurance;
  const healthProfile = profile?.health;

const formatShapFeature = (feature) => {
  const names = {
    HbA1c_level: "HbA1c",
    blood_glucose_level: "Blood Glucose",
    age_years: "Age",
    age: "Age",
    ap_hi: "Systolic BP",
    ap_lo: "Diastolic BP",
    bmi: "BMI",
    cholesterol: "Cholesterol",
    hypertension: "Hypertension",
    smoking_history: "Smoking",
    active: "Physical Activity",
  };

  return names[feature] || feature;
};

const shapFeatures = health?.shap_top_features?.slice(0, 5) || [];
const monthlyIncome = Number(finance?.monthly_income ?? profile?.finance?.income ?? 0);

const fixedExpenses = Number(profile?.finance?.fixed_expenses ?? 0);

const variableExpenses = Number(profile?.finance?.variable_expenses ?? 0);

const monthlyEmis = Number(profile?.finance?.emis ?? 0);

const monthlyExpenses = fixedExpenses + variableExpenses;

const disposableIncome =
  monthlyIncome - monthlyExpenses - monthlyEmis;

const maxShapImpact = Math.max(
  ...shapFeatures.map((item) => Math.abs(Number(item.impact || 0))),
  1
);
  const simulation = data?.simulation;
  const toolTrace = simulation?.tool_trace || [];

  const affordabilityTool = toolTrace.find(
    (tool) => tool.tool === "check_affordability"
  );

  const coverageTool = toolTrace.find(
    (tool) => tool.tool === "check_insurance_coverage"
  );
  const healthContextTool = toolTrace.find(
  (tool) => tool.tool === "get_health_context"
);

const recommendationTool = toolTrace.find(
  (tool) => tool.tool === "get_cheaper_alternative"
);

const healthRisk = healthContextTool?.result?.risk_level;
const urgency = healthContextTool?.result?.urgency;
const agentRecommendation =
  recommendationTool?.result?.recommendation ||
  recommendationTool?.result?.message ||
  recommendationTool?.result?.alternative;

  const treatmentCost = Number(
    affordabilityTool?.result?.cost ??
      affordabilityTool?.input?.cost ??
      0
  );

  const affordable = affordabilityTool?.result?.affordable;
  const coveredAmount = Number(
    coverageTool?.result?.covered_amount ?? 0
  );
  const coverageGap = Number(coverageTool?.result?.gap ?? 0);

  return (
    <div className="app">
      <header className="header">
        <h1>TwinLife AI</h1>
        <p>Health • Finance • Insurance • AI Digital Twin</p>
      </header>

      <main className="container">

        <div className="card">
          <h2>Your Health Profile</h2>

          <div className="profile-grid">
            <div>
              <label>Age</label>
              <input
                type="number"
                value={profile.health.age}
                onChange={(e) =>
                  updateHealth("age", Number(e.target.value))
                }
              />
            </div>

            <div>
              <label>Gender</label>
              <select
                value={profile.health.gender}
                onChange={(e) =>
                  updateHealth("gender", e.target.value)
                }
              >
                <option value="male">Male</option>
                <option value="female">Female</option>
              </select>
            </div>

            <div>
              <label>Height (cm)</label>
              <input
                type="number"
                value={profile.health.height_cm}
                onChange={(e) =>
                  updateHealth("height_cm", Number(e.target.value))
                }
              />
            </div>

            <div>
              <label>Weight (kg)</label>
              <input
                type="number"
                value={profile.health.weight_kg}
                onChange={(e) =>
                  updateHealth("weight_kg", Number(e.target.value))
                }
              />
            </div>

            <div>
              <label>Systolic BP</label>
              <input
                type="number"
                value={profile.health.ap_hi}
                onChange={(e) =>
                  updateHealth("ap_hi", Number(e.target.value))
                }
              />
            </div>

            <div>
              <label>Diastolic BP</label>
              <input
                type="number"
                value={profile.health.ap_lo}
                onChange={(e) =>
                  updateHealth("ap_lo", Number(e.target.value))
                }
              />
            </div>

            <div>
              <label>HbA1c</label>
              <input
                type="number"
                step="0.1"
                value={profile.health.HbA1c_level}
                onChange={(e) =>
                  updateHealth("HbA1c_level", Number(e.target.value))
                }
              />
            </div>

            <div>
              <label>Blood Glucose</label>
              <input
                type="number"
                value={profile.health.blood_glucose_level}
                onChange={(e) =>
                  updateHealth(
                    "blood_glucose_level",
                    Number(e.target.value)
                  )
                }
              />
            </div>
            <div>
  <label>Cholesterol</label>
  <select
    value={profile.health.cholesterol}
    onChange={(e) =>
      updateHealth("cholesterol", Number(e.target.value))
    }
  >
    <option value={1}>Normal</option>
    <option value={2}>Above Normal</option>
    <option value={3}>Well Above Normal</option>
  </select>
</div>

<div>
  <label>Glucose Level</label>
  <select
    value={profile.health.gluc}
    onChange={(e) =>
      updateHealth("gluc", Number(e.target.value))
    }
  >
    <option value={1}>Normal</option>
    <option value={2}>Above Normal</option>
    <option value={3}>Well Above Normal</option>
  </select>
</div>

<div>
  <label>Alcohol Consumption</label>
  <select
    value={profile.health.alco}
    onChange={(e) =>
      updateHealth("alco", Number(e.target.value))
    }
  >
    <option value={0}>No</option>
    <option value={1}>Yes</option>
  </select>
</div>

<div>
  <label>Hypertension</label>
  <select
    value={profile.health.hypertension}
    onChange={(e) =>
      updateHealth("hypertension", Number(e.target.value))
    }
  >
    <option value={0}>No</option>
    <option value={1}>Yes</option>
  </select>
</div>

<div>
  <label>Heart Disease</label>
  <select
    value={profile.health.heart_disease}
    onChange={(e) =>
      updateHealth("heart_disease", Number(e.target.value))
    }
  >
    <option value={0}>No</option>
    <option value={1}>Yes</option>
  </select>
</div>

            <div>
              <label>Smoking</label>
              <select
                value={profile.health.smoking_history}
                onChange={(e) => {
                  const value = e.target.value;

                  updateHealth("smoking_history", value);
                  updateHealth(
                    "smoke",
                    value === "current" ? 1 : 0
                  );
              }}
              >
                <option value="never">Never</option>
                <option value="former">Former</option>
                <option value="current">Current</option>
              </select>
            </div>

            <div>
              <label>Physical Activity</label>
              <select
                value={profile.health.active}
                onChange={(e) =>
                  updateHealth("active", Number(e.target.value))
                }
              >
                <option value={1}>Active</option>
                <option value={0}>Inactive</option>
              </select>
            </div>
          </div>
        </div>

        <div className="card">
          <h2>Your Financial Profile</h2>

          <div className="profile-grid">
            <div>
              <label>Monthly Income (Rs.)</label>
              <input
                type="number"
                value={profile.finance.income}
                onChange={(e) =>
                  updateFinance("income", Number(e.target.value))
                }
              />
            </div>

            <div>
              <label>Fixed Expenses (Rs.)</label>
              <input
                type="number"
                value={profile.finance.fixed_expenses}
                onChange={(e) =>
                  updateFinance(
                    "fixed_expenses",
                    Number(e.target.value)
                  )
                }
              />
            </div>

            <div>
              <label>Variable Expenses (Rs.)</label>
              <input
                type="number"
                value={profile.finance.variable_expenses}
                onChange={(e) =>
                  updateFinance(
                    "variable_expenses",
                    Number(e.target.value)
                  )
                }
              />
            </div>

            <div>
              <label>EMIs (Rs.)</label>
              <input
                type="number"
                value={profile.finance.emis}
                onChange={(e) =>
                  updateFinance("emis", Number(e.target.value))
                }
              />
            </div>

            <div>
              <label>Savings Balance (Rs.)</label>
              <input
                type="number"
                value={profile.finance.savings_balance}
                onChange={(e) =>
                  updateFinance(
                    "savings_balance",
                    Number(e.target.value)
                  )
                }
              />
            </div>
          </div>
        </div>

        <div className="card">
          <h2>Your Insurance Profile</h2>

          <div className="profile-grid">
            <div>
              <label>Sum Insured (Rs.)</label>
              <input
                type="number"
                value={profile.insurance.sum_insured}
                onChange={(e) =>
                  updateInsurance(
                    "sum_insured",
                    Number(e.target.value)
                  )
                }
              />
            </div>

            <div>
              <label>Annual Premium (Rs.)</label>
              <input
                type="number"
                value={profile.insurance.annual_premium}
                onChange={(e) =>
                  updateInsurance(
                    "annual_premium",
                    Number(e.target.value)
                  )
                }
              />
            </div>

            <div>
              <label>Annual Income (Rs.)</label>
              <input
                type="number"
                value={profile.insurance.annual_income}
                onChange={(e) =>
                  updateInsurance(
                    "annual_income",
                    Number(e.target.value)
                  )
                }
              />
            </div>
          </div>
        </div>

        <div className="card">
          <h2>Ask TwinLife</h2>

          <p className="subtitle">
            Ask questions about your health, finances, insurance, or treatment
            affordability.
          </p>

          <textarea
            placeholder="Example: Can I afford a Rs.5 lakh surgery?"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
          />

          <div className="quick-questions">
            <button
              className="quick-button"
              onClick={() => setQuery("What is my health status?")}
            >
              Health Status
            </button>

            <button
              className="quick-button"
              onClick={() =>
                setQuery("Can I afford a Rs.5 lakh surgery?")
              }
            >
              Surgery Affordability
            </button>

            <button
              className="quick-button"
              onClick={() =>
                setQuery("What is my insurance coverage?")
              }
            >
              Insurance Coverage
            </button>

            <button
              className="quick-button"
              onClick={() =>
                setQuery("How is my financial situation?")
              }
            >
              Financial Health
            </button>
          </div>

          <button onClick={askTwinLife} disabled={loading}>
  {loading ? "Analyzing your digital twin..." : "Ask TwinLife"}
</button>
{loading && (
  <div className="loading-status">
    <div className="loading-spinner"></div>
    <p>Analyzing your digital twin...</p>

    <div className="loading-steps">
      <span>Health</span>
      <span>Finance</span>
      <span>Insurance</span>
      <span>RAG</span>
      <span>AI Analysis</span>
    </div>
  </div>
)}
        </div>

        {data?.error && (
          <div className="card error">
            {data.error}
          </div>
        )}

        {data && !data.error && (
          <>
            {simulation && treatmentCost > 0 && (
              <div className="card treatment-summary">
                <h2>Surgery Affordability Summary</h2>

                <div className="summary-grid">
                  <div>
                    <span>Treatment Cost</span>
                    <strong>
                      Rs. {treatmentCost.toLocaleString("en-IN")}
                    </strong>
                  </div>

                  <div>
                    <span>Insurance Covered</span>
                    <strong>
                      Rs. {coveredAmount.toLocaleString("en-IN")}
                    </strong>
                  </div>

                  <div>
                    <span>Coverage Gap</span>
                    <strong>
                      Rs. {coverageGap.toLocaleString("en-IN")}
                    </strong>
                  </div>

                 <div>
  <span>Self-Funding</span>
  <strong>
    {affordable ? "Affordable" : "Not Affordable"}
  </strong>
</div>

<div>
  <span>Final Patient Cost</span>
  <strong>
    Rs. {coverageGap.toLocaleString("en-IN")}
  </strong>
</div> 
                </div>
              </div>
            )}

            <div className="dashboard">
              {health && (
  <div className="result-card health-card">
    <h2>Health</h2>

    <div className="score">
      {health.overall_health_score}/100
    </div>

    <p>Overall Health Score</p>

    <div className="details">
      <p>
        <strong>BMI:</strong> {health.bmi}
      </p>

      <p>
        <strong>BMI Category:</strong>{" "}
        {health.obesity_class}
      </p>

      <p>
        <strong>Blood Pressure:</strong>{" "}
        {healthProfile?.ap_hi}/{healthProfile?.ap_lo} mmHg
      </p>

      <p>
        <strong>BP Category:</strong>{" "}
        {health.hypertension_stage}
      </p>

      <p>
        <strong>HbA1c:</strong>{" "}
        {healthProfile?.HbA1c_level}%
      </p>

      <p>
        <strong>Blood Glucose:</strong>{" "}
        {healthProfile?.blood_glucose_level} mg/dL
      </p>

      <p>
        <strong>Cardiovascular Risk:</strong>{" "}
        {health.cardio_risk?.label}
      </p>

      <p>
        <strong>Diabetes Risk:</strong>{" "}
        {health.diabetes_risk?.label}
      </p>

      <p>
        <strong>Smoking:</strong>{" "}
        {healthProfile?.smoking_history}
      </p>

      <p>
        <strong>Physical Activity:</strong>{" "}
        {healthProfile?.active ? "Active" : "Inactive"}
      </p>
    </div>

    {shapFeatures.length > 0 && (
      <div className="shap-section">
        <h3>Top Contributing Factors</h3>

        {shapFeatures.map((item, index) => {
          const impact = Math.abs(Number(item.impact || 0));
          const width = (impact / maxShapImpact) * 100;

          return (
            <div className="shap-row" key={index}>
              <div className="shap-label">
                {formatShapFeature(item.feature)}
              </div>

              <div className="shap-bar-container">
                <div
                  className="shap-bar"
                  style={{ width: `${width}%` }}
                />
              </div>

              <div className="shap-value">
                {impact.toFixed(3)}
              </div>
            </div>
          );
        })}
      </div>
    )}
  </div>
)}

              {finance && (
  <div className="result-card finance-card">
    <h2>Finance</h2>

    <div className="score">
      {finance.financial_stability_score}/100
    </div>

    <p>Financial Stability Score</p>

    <div className="finance-summary">
      <div className="finance-metric">
        <span>Monthly Income</span>
        <strong>
          Rs. {monthlyIncome.toLocaleString("en-IN")}
        </strong>
      </div>

      <div className="finance-metric">
        <span>Monthly Expenses</span>
        <strong>
          Rs. {monthlyExpenses.toLocaleString("en-IN")}
        </strong>
      </div>

      <div className="finance-metric">
        <span>EMIs</span>
        <strong>
          Rs. {monthlyEmis.toLocaleString("en-IN")}
        </strong>
      </div>

      <div className="finance-metric">
        <span>Disposable Income</span>
        <strong>
          Rs. {disposableIncome.toLocaleString("en-IN")}
        </strong>
      </div>
    </div>

    <div className="details">
      <p>
        <strong>DTI Ratio:</strong>{" "}
        {finance.dti_ratio}
      </p>

      <p>
        <strong>Savings Rate:</strong>{" "}
        {finance.savings_rate}
      </p>

      <p>
        <strong>Emergency Fund:</strong>{" "}
        {finance.emergency_fund_months} months
      </p>
    </div>

    <div className="finance-breakdown">
      <h3>Monthly Financial Breakdown</h3>

      <div className="breakdown-row">
        <span>Income</span>
        <strong>
          Rs. {monthlyIncome.toLocaleString("en-IN")}
        </strong>
      </div>

      <div className="breakdown-row">
        <span>Fixed Expenses</span>
        <strong>
          Rs. {fixedExpenses.toLocaleString("en-IN")}
        </strong>
      </div>

      <div className="breakdown-row">
        <span>Variable Expenses</span>
        <strong>
          Rs. {variableExpenses.toLocaleString("en-IN")}
        </strong>
      </div>

      <div className="breakdown-row">
        <span>EMIs</span>
        <strong>
          Rs. {monthlyEmis.toLocaleString("en-IN")}
        </strong>
      </div>

      <div className="breakdown-divider" />

      <div className="breakdown-row disposable">
        <span>Disposable Income</span>
        <strong>
          Rs. {disposableIncome.toLocaleString("en-IN")}
        </strong>
      </div>
    </div>
  </div>
)}

              {insurance && (
  <div className="result-card insurance-card">
    <h2>Insurance</h2>

    <div className="score">
      {insurance.insurance_adequacy_score}/100
    </div>

    <p>Overall Insurance Adequacy</p>

    <div className="insurance-summary">
      <div className="insurance-metric">
        <span>Sum Insured</span>
        <strong>
          Rs. {Number(
            profile.insurance.sum_insured || 0
          ).toLocaleString("en-IN")}
        </strong>
      </div>

      <div className="insurance-metric">
        <span>Annual Premium</span>
        <strong>
          Rs. {Number(
            profile.insurance.annual_premium || 0
          ).toLocaleString("en-IN")}
        </strong>
      </div>

      <div className="insurance-metric">
        <span>Annual Income</span>
        <strong>
          Rs. {Number(
            profile.insurance.annual_income || 0
          ).toLocaleString("en-IN")}
        </strong>
      </div>

      <div className="insurance-metric">
        <span>Rider Gaps</span>
        <strong>
          {insurance.rider_analysis?.gaps?.length ?? 0}
        </strong>
      </div>
    </div>

    <div className="details">
      <p>
        <strong>Coverage Ratio:</strong>{" "}
        {insurance.coverage_adequacy_ratio}
      </p>

      <p>
        <strong>Premium Ratio:</strong>{" "}
        {insurance.premium_affordability_ratio}
      </p>

      <p>
        <strong>Insurance Score:</strong>{" "}
        {insurance.insurance_adequacy_score}/100
      </p>
    </div>

    <div className="insurance-section">
      <h3>Overall Insurance Adequacy</h3>

      <p>
        This score represents the adequacy of your overall
        insurance profile based on coverage, premium
        affordability, and identified gaps.
      </p>
    </div>

    {insurance.rider_analysis?.gaps?.length > 0 && (
      <div className="insurance-section">
        <h3>Identified Rider Gaps</h3>

        <ul className="rider-list">
          {insurance.rider_analysis.gaps.map((gap, index) => (
            <li key={index}>{gap}</li>
          ))}
        </ul>
      </div>
    )}
  </div>
)}
            </div>

            {data.simulation && (
              <div className="card simulation-card">
                <h2>Treatment Simulation</h2>

                {data.simulation.tool_trace?.map((tool, index) => {
                  const result = tool.result || {};

                  if (tool.tool === "check_affordability") {
                    return (
                      <div
                        className="simulation-section"
                        key={index}
                      >
                        <h3>Financial Affordability</h3>

                        <p>
                          <strong>Treatment Cost:</strong>{" "}
                          Rs.{" "}
                          {Number(
                            result.cost ??
                              tool.input?.cost ??
                              0
                          ).toLocaleString("en-IN")}
                        </p>

                        <p>
                          <strong>Affordable:</strong>{" "}
                          {result.affordable ? "Yes" : "No"}
                        </p>

                        <p>
                          <strong>Affordability Score:</strong>{" "}
                          {result.score}
                        </p>
                      </div>
                    );
                  }

                  if (
                    tool.tool ===
                    "check_insurance_coverage"
                  ) {
                    return (
                      <div
                        className="simulation-section"
                        key={index}
                      >
                        <h3>Insurance Coverage</h3>

                        <p>
                          <strong>Covered Amount:</strong>{" "}
                          Rs.{" "}
                          {Number(
                            result.covered_amount || 0
                          ).toLocaleString("en-IN")}
                        </p>

                        <p>
                          <strong>Coverage Gap:</strong>{" "}
                          Rs.{" "}
                          {Number(
                            result.gap || 0
                          ).toLocaleString("en-IN")}
                        </p>
                      </div>
                    );
                  }

                  if (
                    tool.tool ===
                    "get_health_context"
                  ) {
                    return (
                      <div
                        className="simulation-section"
                        key={index}
                      >
                        <h3>Health Context</h3>

                        <p>
                          <strong>Risk Level:</strong>{" "}
                          {result.risk_level}
                        </p>

                        <p>
                          <strong>Urgency:</strong>{" "}
                          {result.urgency}
                        </p>
                      </div>
                    );
                  }
                  if (
  tool.tool ===
  "get_cheaper_alternative"
) {
  return (
    <div
      className="simulation-section"
      key={index}
    >
      <h3>Agent Recommendation</h3>

      <p>
        {result.recommendation ||
          result.message ||
          result.alternative ||
          "No recommendation available."}
      </p>
    </div>
  );
}


                  return null;
                })}
              </div>
            )}

            

            {data.explanation && (
              <div className="card response-card">
                <h2>AI Explanation</h2>

                <div className="response">
                  {data.explanation}
                </div>
              </div>
            )}

            {data.rag && data.rag.length > 0 && (
  <div className="card rag-card">
    <h2>📚 Guidelines Used</h2>

    <div className="guideline-list">
      {data.rag.map((chunk, index) => {
        let source = "General Wellness Guidelines";

        if (
          chunk.includes("FRAMINGHAM") ||
          chunk.includes("HEART") ||
          chunk.includes("cardiovascular")
        ) {
          source = "Cardiovascular Guidelines";
        } else if (
          chunk.includes("DIABETES") ||
          chunk.includes("HbA1c") ||
          chunk.includes("blood glucose")
        ) {
          source = "Diabetes Management Guidelines";
        } else if (
          chunk.includes("GENERAL WELLNESS") ||
          chunk.includes("WHO") ||
          chunk.includes("ICMR")
        ) {
          source = "General Wellness Guidelines";
        }

        return (
          <div className="guideline-item" key={index}>
            <span>•</span>
            <strong>{source}</strong>
          </div>
        );
      })}
    </div>

    <div className="rag-section">
      <h3>Retrieved Knowledge</h3>
      <p className="rag-description">
        Top {Math.min(data.rag.length, 3)} relevant guideline chunks
        retrieved by the RAG system.
      </p>

      {data.rag.slice(0, 3).map((chunk, index) => (
        <div className="rag-chunk" key={index}>
          <div className="rag-chunk-title">
            Guideline Chunk {index + 1}
          </div>

          <p>{chunk}</p>
        </div>
      ))}
    </div>
  </div>
)}
          </>
        )}
      </main>
    </div>
  );
}

export default App;