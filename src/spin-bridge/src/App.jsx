import { useState } from "react";
import { BrowserRouter, Routes, Route } from "react-router-dom";
import Sidebar from "./ui/Sidebar";
import Home from "./pages/Home";
import Agents from "./pages/Agents";
import Conversations from "./pages/Conversations";
import { ThemeToggle } from "./ui/ThemeToggle"; // <--- Importa el botón
import { useDarkMode } from "./hooks/useDarkMode";

function App() {
  const [isDark, setIsDark] = useDarkMode();
  const [days, setDays] = useState(7);

  return (
    <BrowserRouter>
      {/* Añadimos 'selection:bg-brand-primary' para un toque pro y 
        aseguramos que el fondo cubra todo el viewport.
      */}
      <div className="flex bg-gray-50 dark:bg-slate-800 min-h-screen transition-colors duration-300 selection:bg-brand-primary/30">
        <Sidebar />

        {/* Cambiamos 'relative' por un contenedor que gestione mejor el scroll 
          y evite que el contenido se desplace sobre el fondo del HTML.
        */}
        <main className="ml-64 flex-1 flex flex-col min-h-screen">
          <div className="fixed top-6 right-8 z-50">
            <ThemeToggle isDark={isDark} setIsDark={setIsDark} />
          </div>

          {/* Contenedor de rutas con padding para que nada quede oculto tras el toggle */}
          <div className="flex-1">
            <Routes>
              <Route
                path="/"
                element={<Home days={days} setDays={setDays} />}
              />
              <Route path="/config" element={<Agents />} />
              <Route
                path="/chats"
                element={<Conversations days={days} setDays={setDays} />}
              />
            </Routes>
          </div>
        </main>
      </div>
    </BrowserRouter>
  );
}

export default App;
