import { useState, useEffect, useCallback } from "react";
import axios from "axios";

export const useApi = (endpoint) => {
  const [data, setData] = useState(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState(null);

  const fetchData = useCallback(async () => {
    if (!endpoint) return;
    setIsLoading(true);
    try {
      // Usamos la IP explícita y el puerto 8000 de FastAPI
      const cleanEndpoint = endpoint.trim();
      const response = await axios.get(`http://127.0.0.1:8000${cleanEndpoint}`);
      setData(response.data);
      setError(null);
    } catch (err) {
      // Manejo amigable de errores de conexión
      let errorMessage = "Ocurrió un error inesperado.";

      if (err.code === "ERR_CONNECTION_REFUSED") {
        errorMessage =
          "No se pudo conectar al servidor. ¿Está encendido el Backend (uvicorn)?";
      } else if (err.response) {
        // El servidor respondió con un error (4xx o 5xx)
        errorMessage = `Error del servidor: ${err.response.status}`;
      } else {
        errorMessage = err.message;
      }

      setError(errorMessage);
      console.error("API Error:", err);
    } finally {
      setIsLoading(false);
    }
  }, [endpoint]);

  // Se ejecuta cada vez que el endpoint cambia (ej: al cambiar de 7 a 30 días)
  useEffect(() => {
    fetchData();
  }, [fetchData]);

  return { data, isLoading, error, refetch: fetchData };
};
