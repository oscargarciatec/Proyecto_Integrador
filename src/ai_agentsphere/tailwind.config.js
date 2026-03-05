/** @type {import('tailwindcss').Config} */
export default {
  // 1. ESTA ES LA LÍNEA CLAVE. Sin esto, 'dark:' no funciona con clases.
  darkMode: "class",
  content: ["./index.html", "./src/**/*.{js,ts,jsx,tsx}"],
  theme: {
    extend: {
      colors: {
        brand: {
          primary: "#00C2CB",
          dark: "#1E293B",
        },
      },
      fontFamily: {
        montserrat: ["Montserrat", "sans-serif"],
        poppins: ["Poppins", "sans-serif"],
      },
    },
  },
  plugins: [],
};
