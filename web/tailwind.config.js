/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{js,ts,tsx}"],
  theme: {
    extend: {
      fontFamily: {
        serif: ['"Iowan Old Style"', "Palatino", "Georgia", "serif"],
        sans: ["ui-sans-serif", "system-ui", "sans-serif"],
        mono: ["ui-monospace", "Menlo", "monospace"],
      },
      colors: {
        paper: "#f6f1e8",
        ink: "#1c1917",
        rule: "#d6cbb8",
        flag: "#b45309",
      },
    },
  },
  plugins: [],
};
