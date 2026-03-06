import React, { useEffect, useCallback, useMemo, useReducer } from "react";
import axios from "axios";

const initialState = {
  data: null,
  isLoading: true,
  error: null,
};

function apiReducer(state, action) {
  switch (action.type) {
    case "FETCH_INIT":
      if (state.isLoading && !state.error) return state;
      return { ...state, isLoading: true, error: null };
    case "FETCH_SUCCESS":
      if (
        JSON.stringify(state.data) === JSON.stringify(action.payload) &&
        !state.isLoading
      )
        return state;
      return { ...state, isLoading: false, data: action.payload, error: null };
    case "FETCH_FAILURE":
      return { ...state, isLoading: false, error: action.payload };
    default:
      return state;
  }
}

export const useApi = (endpoint) => {
  const [state, dispatch] = useReducer(apiReducer, initialState);

  const fetchData = useCallback(
    async (signal) => {
      if (!endpoint) return;
      dispatch({ type: "FETCH_INIT" });
      try {
        const cleanEndpoint = endpoint.trim();
        const apiBaseUrl =
          import.meta.env.VITE_API_URL || "http://127.0.0.1:8000";
        const fullUrl = `${apiBaseUrl}${cleanEndpoint}`;

        console.log(`[useApi] Fetching: ${fullUrl}`);

        const response = await axios.get(fullUrl, {
          signal,
        });
        dispatch({ type: "FETCH_SUCCESS", payload: response.data });
      } catch (err) {
        console.error("[useApi] Request failed:", err);
        console.error("[useApi] Error details:", {
          message: err.message,
          code: err.code,
          response: err.response?.data,
          status: err.response?.status,
        });
        if (axios.isCancel(err)) return; // Ignoramos errores de cancelación voluntaria

        let errorMessage = "Ocurrió un error inesperado.";
        if (err.code === "ERR_CONNECTION_REFUSED") {
          errorMessage =
            "No se pudo conectar al servidor. ¿Está encendido el Backend (uvicorn)?";
        } else if (err.response) {
          errorMessage = `Error del servidor: ${err.response.status}`;
        } else {
          errorMessage = err.message;
        }
        dispatch({ type: "FETCH_FAILURE", payload: errorMessage });
        console.error("API Error:", err);
      }
    },
    [endpoint],
  );

  useEffect(() => {
    const controller = new AbortController();
    fetchData(controller.signal);

    return () => {
      controller.abort();
    };
  }, [fetchData, endpoint]);

  return useMemo(() => ({ ...state, refetch: fetchData }), [state, fetchData]);
};
