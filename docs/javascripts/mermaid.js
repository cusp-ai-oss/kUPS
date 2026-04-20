// Mermaid configuration for dark/light mode support
window.mermaidConfig = {
  theme: 'base',
  themeVariables: {
    // Primary colors (blue) - for main processing nodes
    primaryColor: '#2094f3',
    primaryTextColor: '#ffffff',
    primaryBorderColor: '#1976d2',

    // Secondary colors (green) - for outputs and results
    secondaryColor: '#4caf50',
    secondaryTextColor: '#ffffff',
    secondaryBorderColor: '#388e3c',

    // Tertiary colors (orange) - for parameters and storage
    tertiaryColor: '#ff9800',
    tertiaryTextColor: '#ffffff',
    tertiaryBorderColor: '#f57c00',

    // General styling
    mainBkg: '#2094f3',
    noteBkgColor: '#ff9800',
    noteTextColor: '#ffffff',
    noteBorderColor: '#f57c00',

    // Lines and edges
    lineColor: '#666666',

    // Text colors
    textColor: '#ffffff',
    labelTextColor: '#ffffff',

    // Additional colors
    errorBkgColor: '#f44336',
    errorTextColor: '#ffffff',

    // Cluster/subgraph styling
    clusterBkg: 'rgba(128, 128, 128, 0.1)',
    clusterBorder: '#666666',

    // Node specific
    nodeBorder: '#1976d2',
    nodeTextColor: '#ffffff',

    // Edge label background
    edgeLabelBackground: 'rgba(0, 0, 0, 0.7)',

    // Font settings
    fontFamily: 'Inter, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif',
    fontSize: '14px'
  },

  // Flowchart specific settings
  flowchart: {
    htmlLabels: true,
    curve: 'basis',
    nodeSpacing: 50,
    rankSpacing: 50,
    padding: 15
  },

  // Sequence diagram settings
  sequence: {
    actorMargin: 50,
    boxTextMargin: 5,
    noteMargin: 10,
    messageMargin: 35
  },

  // Security
  securityLevel: 'loose',

  // Start on load
  startOnLoad: true
};

// Initialize Mermaid when DOM is ready
if (typeof mermaid !== 'undefined') {
  mermaid.initialize(window.mermaidConfig);
}
