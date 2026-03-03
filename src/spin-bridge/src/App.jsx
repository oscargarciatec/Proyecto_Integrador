import { BrowserRouter, Routes, Route } from "react-router-dom";
import Sidebar from "./ui/Sidebar";
import Home from "./pages/Home";
import Config from "./pages/Config";
import Conversations from "./pages/Conversations";
import { useState } from "react";

function App() {
  const [days, setDays] = useState(7);

  return (
    <BrowserRouter>
      <div className="flex bg-gray-50 min-h-screen">
        <Sidebar />
        <main className="ml-64 flex-1">
          <Routes>
            <Route path="/" element={<Home days={days} setDays={setDays} />} />
            <Route path="/config" element={<Config />} />
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
