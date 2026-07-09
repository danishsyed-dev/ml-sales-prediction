// Theme Toggle Initialization
const themeToggleBtn = document.getElementById("theme-toggle");
const storedTheme = localStorage.getItem("theme");

if (storedTheme === "dark") {
  document.body.classList.add("dark-theme");
} else if (storedTheme === "light") {
  document.body.classList.remove("dark-theme");
} else if (window.matchMedia && window.matchMedia("(prefers-color-scheme: dark)").matches) {
  document.body.classList.add("dark-theme");
}

if (themeToggleBtn) {
  themeToggleBtn.addEventListener("click", () => {
    document.body.classList.toggle("dark-theme");
    const theme = document.body.classList.contains("dark-theme") ? "dark" : "light";
    localStorage.setItem("theme", theme);
  });
}

const form = document.getElementById("predict-form");
const statusEl = document.getElementById("status");
const optimalPriceEl = document.getElementById("optimal_price");
const maxSalesEl = document.getElementById("max_sales");
const rowsReturnedEl = document.getElementById("rows_returned");
const tableBody = document.getElementById("table-body");
const errorBox = document.getElementById("error-box");
const errorList = document.getElementById("error-list");

const chartLine = document.getElementById("chart-line");
const chartPoints = document.getElementById("chart-points");
const chartGrid = document.getElementById("chart-grid");

const formatCurrency = (value) => {
  if (value === null || value === undefined || Number.isNaN(value)) {
    return "--";
  }
  return new Intl.NumberFormat("en-US", {
    style: "currency",
    currency: "USD",
    maximumFractionDigits: 2,
  }).format(value);
};

const formatNumber = (value) => {
  if (value === null || value === undefined || Number.isNaN(value)) {
    return "--";
  }
  return new Intl.NumberFormat("en-US", { maximumFractionDigits: 0 }).format(value);
};

const setStatus = (text, isLoading = false) => {
  statusEl.textContent = text;
  if (isLoading) {
    statusEl.classList.add("status--loading");
  } else {
    statusEl.classList.remove("status--loading");
  }
};

const clearErrors = () => {
  errorList.innerHTML = "";
  errorBox.hidden = true;
  document.querySelectorAll(".field__input.is-invalid").forEach((el) => {
    el.classList.remove("is-invalid");
  });
};

const showErrors = (items) => {
  errorList.innerHTML = items.map((item) => `<li>${item}</li>`).join("");
  errorBox.hidden = false;
};

const validateInputs = ({ stockcode, unitpriceMin, unitpriceMax, numPriceBins }) => {
  const errors = [];

  const stockEl = document.getElementById("stockcode");
  const minEl = document.getElementById("unitprice_min");
  const maxEl = document.getElementById("unitprice_max");
  const binsEl = document.getElementById("num_price_bins");

  if (!stockcode) {
    errors.push("Stock code is required.");
    stockEl.classList.add("is-invalid");
  }

  const minVal = unitpriceMin ? Number(unitpriceMin) : null;
  const maxVal = unitpriceMax ? Number(unitpriceMax) : null;

  if (unitpriceMin && (Number.isNaN(minVal) || minVal < 0)) {
    errors.push("Min unit price must be a non-negative number.");
    minEl.classList.add("is-invalid");
  }

  if (unitpriceMax && (Number.isNaN(maxVal) || maxVal < 0)) {
    errors.push("Max unit price must be a non-negative number.");
    maxEl.classList.add("is-invalid");
  }

  if (minVal !== null && maxVal !== null && minVal > maxVal) {
    errors.push("Min unit price must be less than or equal to max unit price.");
    minEl.classList.add("is-invalid");
    maxEl.classList.add("is-invalid");
  }

  const binsVal = numPriceBins ? Number(numPriceBins) : 100;
  if (Number.isNaN(binsVal) || binsVal < 10 || binsVal > 300) {
    errors.push("Price bins must be a number between 10 and 300.");
    binsEl.classList.add("is-invalid");
  }

  return errors;
};

const renderTable = (rows) => {
  if (!rows.length) {
    tableBody.innerHTML = `
      <tr>
        <td colspan="3" class="table__empty">No rows returned.</td>
      </tr>`;
    return;
  }

  const html = rows
    .map(
      (row) => `
      <tr>
        <td>${formatCurrency(row.unit_price)}</td>
        <td>${formatNumber(row.quantity)}</td>
        <td>${formatCurrency(row.predicted_sales)}</td>
      </tr>`
    )
    .join("");
  tableBody.innerHTML = html;
};

const renderChart = (rows) => {
  chartLine.setAttribute("points", "");
  chartPoints.innerHTML = "";
  chartGrid.innerHTML = "";

  if (!rows.length) {
    return;
  }

  const width = 800;
  const height = 300;
  const padding = 40;

  const prices = rows.map((row) => row.unit_price);
  const sales = rows.map((row) => row.predicted_sales);

  const minX = Math.min(...prices);
  const maxX = Math.max(...prices);
  const minY = Math.min(...sales);
  const maxY = Math.max(...sales);

  const scaleX = (value) =>
    padding + ((value - minX) / (maxX - minX || 1)) * (width - padding * 2);
  const scaleY = (value) =>
    height - padding - ((value - minY) / (maxY - minY || 1)) * (height - padding * 2);

  const gridLines = 5;
  for (let i = 0; i <= gridLines; i += 1) {
    const y = padding + (i / gridLines) * (height - padding * 2);
    const line = document.createElementNS("http://www.w3.org/2000/svg", "line");
    line.setAttribute("x1", padding);
    line.setAttribute("x2", width - padding);
    line.setAttribute("y1", y);
    line.setAttribute("y2", y);
    line.setAttribute("class", "chart__grid");
    chartGrid.appendChild(line);
  }

  const points = rows
    .map((row) => `${scaleX(row.unit_price)},${scaleY(row.predicted_sales)}`)
    .join(" ");
  chartLine.setAttribute("points", points);

  rows.slice(0, 40).forEach((row) => {
    const circle = document.createElementNS("http://www.w3.org/2000/svg", "circle");
    circle.setAttribute("cx", scaleX(row.unit_price));
    circle.setAttribute("cy", scaleY(row.predicted_sales));
    circle.setAttribute("r", "3");
    circle.setAttribute("class", "chart__point");
    chartPoints.appendChild(circle);
  });
};

const updateSummary = (rows) => {
  rowsReturnedEl.textContent = rows.length.toString();

  if (!rows.length) {
    optimalPriceEl.textContent = "--";
    maxSalesEl.textContent = "--";
    return;
  }

  const first = rows[0];
  optimalPriceEl.textContent = formatCurrency(first.optimal_unit_price);
  maxSalesEl.textContent = formatCurrency(first.max_predicted_sales);
};

form.addEventListener("submit", async (event) => {
  event.preventDefault();

  const stockcode = document.getElementById("stockcode").value.trim() || "85123A";
  const unitpriceMin = document.getElementById("unitprice_min").value.trim();
  const unitpriceMax = document.getElementById("unitprice_max").value.trim();
  const numPriceBins = document.getElementById("num_price_bins").value.trim();

  clearErrors();
  const errors = validateInputs({
    stockcode,
    unitpriceMin,
    unitpriceMax,
    numPriceBins,
  });
  if (errors.length) {
    showErrors(errors);
    setStatus("Validation errors");
    return;
  }

  const params = new URLSearchParams();
  if (unitpriceMin) params.set("unitprice_min", unitpriceMin);
  if (unitpriceMax) params.set("unitprice_max", unitpriceMax);
  if (numPriceBins) params.set("num_price_bins", numPriceBins);

  setStatus("Running prediction...", true);
  const submitButton = form.querySelector(".button");
  submitButton.disabled = true;

  try {
    const response = await fetch(`/v1/predict-price/${encodeURIComponent(stockcode)}?${params.toString()}`);
    if (!response.ok) {
      throw new Error(`Request failed with status ${response.status}`);
    }
    const data = await response.json();

    const rows = Array.isArray(data) ? data : [];
    updateSummary(rows);

    const sortedRows = [...rows].sort((a, b) => b.predicted_sales - a.predicted_sales);
    renderTable(sortedRows.slice(0, 10));
    renderChart(rows);

    setStatus(`✓ Complete — ${stockcode}`, false);
  } catch (error) {
    console.error(error);
    setStatus("✗ Error occurred", false);
    showErrors(["The API request failed. Please check the server logs and try again."]);
    updateSummary([]);
    renderTable([]);
    renderChart([]);
  } finally {
    submitButton.disabled = false;
  }
});
