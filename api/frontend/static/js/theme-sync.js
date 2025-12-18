// Listen for theme updates from the parent
window.addEventListener("message", (event) => {
    if (event.data?.type !== "theme-update") return;

    const theme = event.data.theme;

    // Apply theme to the iframe <html> element
    document.documentElement.setAttribute("data-theme", theme);

    // If page contains a Plotly figure, recolor it
    if (window.__PLOT_DIV_ID__) {
        applyPlotlyTheme(window.__PLOT_DIV_ID__, theme);
    }
});

// Default theme if iframe loads before parent sends message
document.documentElement.setAttribute("data-theme", "light");


// ---- Plotly theme switcher ----
function applyPlotlyTheme(divId, theme) {
    const darkMode = (theme === "dark");

    const gd = document.getElementById(divId);

    const xType = gd.layout?.xaxis?.type || "linear";
    const yType = gd.layout?.yaxis?.type || "linear";

    const update = {
        paper_bgcolor: darkMode ? "black" : "white",
        plot_bgcolor: darkMode ? "black" : "white",
        font: { color: darkMode ? "white" : "black" },
        xaxis: {
            type: xType,
            gridcolor: darkMode ? "#444" : "#ccc",
            zerolinecolor: darkMode ? "#888" : "#444"
        },
        yaxis: {
            type: yType,
            gridcolor: darkMode ? "#444" : "#ccc",
            zerolinecolor: darkMode ? "#888" : "#444"
        },
        legend: {
            font: {color: darkMode ? 'white': "black"}
        }
    };

    Plotly.relayout(divId, update);
}
