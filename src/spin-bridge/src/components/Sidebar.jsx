import { MessageSquare, Settings, LayoutDashboard } from "lucide-react";
import { Link } from "react-router-dom";

export default function Sidebar() {
  return (
    // Sidebar con color Primario y fuente Montserrat
    <div className="w-64 h-screen bg-brand-white text-brand-dark p-5 flex flex-col fixed font-montserrat">
      <h1 className="text-xl font-bold mb-10 text-brand-primary">
        Spin Bridge
      </h1>
      <nav className="flex flex-col gap-4 font-semibold">
        <Link
          to="/"
          className="flex items-center gap-2 hover:text-brand-orange transition-colors"
        >
          <LayoutDashboard size={20} /> Dashboard
        </Link>
        <Link
          to="/chats"
          className="flex items-center gap-2 hover:text-brand-orange transition-colors"
        >
          <MessageSquare size={20} /> Chats
        </Link>
        <Link
          to="/config"
          className="flex items-center gap-2 hover:text-brand-orange transition-colors"
        >
          <Settings size={20} /> Configuración
        </Link>
      </nav>
    </div>
  );
}
