/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{js,jsx,ts,tsx}"],
  theme: {
    extend: {
      colors: {
        // All colors reference CSS variables so themes swap at runtime.
        // CSS vars hold space-separated RGB triplets (no rgb() wrapper).
        paper: {
          DEFAULT: "rgb(var(--paper) / <alpha-value>)",
          soft:    "rgb(var(--paper-soft) / <alpha-value>)",
          warm:    "rgb(var(--paper-warm) / <alpha-value>)",
          edge:    "rgb(var(--paper-edge) / <alpha-value>)",
          rule:    "rgb(var(--paper-rule) / <alpha-value>)",
        },
        ink: {
          DEFAULT: "rgb(var(--ink) / <alpha-value>)",
          soft:    "rgb(var(--ink-soft) / <alpha-value>)",
          muted:   "rgb(var(--ink-muted) / <alpha-value>)",
          quiet:   "rgb(var(--ink-quiet) / <alpha-value>)",
          faded:   "rgb(var(--ink-faded) / <alpha-value>)",
          ghost:   "rgb(var(--ink-ghost) / <alpha-value>)",
          whisper: "rgb(var(--ink-whisper) / <alpha-value>)",
        },
        accent: {
          DEFAULT: "rgb(var(--accent) / <alpha-value>)",
          soft:    "rgb(var(--accent-soft) / <alpha-value>)",
        },
        category: {
          equation: "#6B5F8C",
          diagram:  "#8B6B4A",
          chart:    "#8C5A47",
        },
      },
      fontFamily: {
        serif: ["ui-serif", '"New York"', "Georgia", "serif"],
        sans: [
          "-apple-system",
          "BlinkMacSystemFont",
          '"SF Pro Text"',
          '"Helvetica Neue"',
          "sans-serif",
        ],
        mono: ['"JetBrains Mono"', '"SF Mono"', "Menlo", "Consolas", "monospace"],
      },
      fontSize: {
        eyebrow:    ["11px", { letterSpacing: "0.18em", lineHeight: "1.4" }],
        "eyebrow-sm": ["10px", { letterSpacing: "0.15em", lineHeight: "1.4" }],
      },
      letterSpacing: {
        editorial: "-0.02em",
      },
      borderRadius: {
        editorial: "3px",
      },
    },
  },
  plugins: [],
};
