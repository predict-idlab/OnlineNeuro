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

    const update = {
        paper_bgcolor: darkMode ? "black" : "white",
        plot_bgcolor: darkMode ? "black" : "white",
        font: { color: darkMode ? "white" : "black" },
        xaxis: {
            gridcolor: darkMode ? "#444" : "#ccc",
            zerolinecolor: darkMode ? "#888" : "#444"
        },
        yaxis: {
            gridcolor: darkMode ? "#444" : "#ccc",
            zerolinecolor: darkMode ? "#888" : "#444"
        }
    };

    Plotly.relayout(divId, update);
}
