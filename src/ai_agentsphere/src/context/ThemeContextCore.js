import { createContext, useContext } from "react";

export const ThemeStateContext = createContext();
export const ThemeDispatchContext = createContext();

export const useThemeState = () => {
  const context = useContext(ThemeStateContext);
  if (context === undefined) {
    throw new Error("useThemeState must be used within a ThemeProvider");
  }
  return context;
};

export const useThemeDispatch = () => {
  const context = useContext(ThemeDispatchContext);
  if (context === undefined) {
    throw new Error("useThemeDispatch must be used within a ThemeProvider");
  }
  return context;
};
