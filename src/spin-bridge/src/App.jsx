import { useState } from "react";
import { BrowserRouter, Routes, Route } from "react-router-dom";
import Sidebar from "./ui/Sidebar";
import Home from "./pages/Home";
import Agents from "./pages/Agents";
import Conversations from "./pages/Conversations";
import { ThemeToggle } from "./ui/ThemeToggle"; // <--- Importa el botón

function App() {
  const [days, setDays] = useState(7);

  return (
    <BrowserRouter>
      <div className="flex bg-gray-50 dark:bg-slate-950 min-h-screen transition-colors duration-300">
        <Sidebar />

        <main className="ml-64 flex-1 relative">
          {/* BOTÓN FLOTANTE: 
              Lo ponemos aquí para que siempre esté arriba a la derecha 
              del contenido principal.
          */}
          <div className="fixed top-6 right-8 z-50">
            <ThemeToggle />
          </div>

          <Routes>
            <Route path="/" element={<Home days={days} setDays={setDays} />} />
            <Route path="/config" element={<Agents />} />
            <Route
              path="/chats"
              element={<Conversations days={days} setDays={setDays} />}
            />
          </Routes>
        </main>
      </div>
    </BrowserRouter>
  );
}

export default App;
