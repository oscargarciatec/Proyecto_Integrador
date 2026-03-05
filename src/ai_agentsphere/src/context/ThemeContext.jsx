import React, { useEffect, useState, useCallback, useMemo } from "react";
import { ThemeStateContext, ThemeDispatchContext } from "./ThemeContextCore";

export const ThemeProvider = ({ children }) => {
  const [isDark, setIsDark] = useState(() => {
    if (typeof window !== "undefined") {
      return (
        localStorage.getItem("theme") === "dark" ||
        (!localStorage.getItem("theme") &&
          window.matchMedia("(prefers-color-scheme: dark)").matches)
      );
    }
    return false;
  });

  useEffect(() => {
    const root = window.document.documentElement;
    if (isDark) {
      root.classList.add("dark");
      localStorage.setItem("theme", "dark");
    } else {
      root.classList.remove("dark");
      localStorage.setItem("theme", "light");
    }
  }, [isDark]);

  console.log("🎨 ThemeProvider render, isDark:", isDark);
  const toggleTheme = useCallback(() => setIsDark((prev) => !prev), []);
  const dispatchValue = useMemo(
    () => ({ toggleTheme, setIsDark }),
    [toggleTheme],
  );

  return (
    <ThemeStateContext.Provider value={isDark}>
      <ThemeDispatchContext.Provider value={dispatchValue}>
        {children}
      </ThemeDispatchContext.Provider>
    </ThemeStateContext.Provider>
  );
};
