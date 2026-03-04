/** @type {import('tailwindcss').Config} */
export default {
  darkMode: "class", // <--- Esta es la clave
  content: ["./index.html", "./src/**/*.{js,ts,jsx,tsx}"],
  theme: {
    extend: {
      colors: {
        brand: {
          primary: "#00C2CB",
          orange: "#FF6F00",
          dark: "#1E293B", // Color para el texto en modo claro o fondos en modo oscuro
        },
      },
    },
  },
  plugins: [],
};
