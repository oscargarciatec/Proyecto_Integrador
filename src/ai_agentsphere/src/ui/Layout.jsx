import Sidebar from "./Sidebar";
import { ThemeToggle } from "./ThemeToggle";

export const Layout = ({ children }) => {
  return (
    <div className="flex min-h-screen bg-[#F8FAFC] dark:bg-slate-950 transition-colors duration-300">
      {/* Sidebar Fijo */}
      <Sidebar />

      {/* Contenedor del Contenido */}
      <main className="flex-1 ml-64 relative">
        {/* Botón Flotante en la esquina superior derecha */}
        <div className="fixed top-6 right-8 z-50">
          <ThemeToggle />
        </div>

        {/* Aquí se renderizan Home, Conversations o Config */}
        <div className="p-4">{children}</div>
      </main>
    </div>
  );
};
