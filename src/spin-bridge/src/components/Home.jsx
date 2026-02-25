import { useState, useEffect } from "react";
import axios from "axios";

const Home = () => {
  const [stats, setStats] = useState({
    total_conversations: 0,
    feedback_percentage: 0,
    avg_response_time: "0s",
    fallback_rate: "0%",
  });

  useEffect(() => {
    const fetchStats = async () => {
      try {
        // Apuntamos al backend de FastAPI
        const response = await axios.get(
          "http://127.0.0.1:8000/api/dashboard/stats",
        );
        setStats(response.data);
      } catch (error) {
        console.error("Error conectando con AlloyDB:", error);
      }
    };
    fetchStats();
  }, []);

  return (
    <div className="p-8 space-y-8 bg-[#F8FAFC] min-h-screen font-work-sans">
      <header className="border-b border-brand-primary/20 pb-4">
        <h2 className="text-4xl font-montserrat font-bold text-brand-dark">
          Dashboard Ejecutivo
        </h2>
        <p className="text-slate-500 font-work-sans">
          Monitoreo de{" "}
          <span className="text-brand-primary font-semibold">Spin Compass</span>{" "}
          en tiempo real
        </p>
      </header>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        {/* Card Primaria (50% color marca) */}
        <div className="bg-brand-primary p-6 rounded-2xl shadow-lg shadow-brand-primary/20 text-white">
          <p className="font-work-sans text-xs uppercase tracking-wider opacity-90">
            Total Conversaciones
          </p>
          <p className="font-poppins font-bold text-4xl mt-2">
            {stats.total_conversations.toLocaleString()}
          </p>
          <div className="mt-4 h-1 w-full bg-white/30 rounded-full overflow-hidden">
            <div className="bg-white h-full w-full animate-pulse"></div>
          </div>
        </div>

        {/* Card Feedback (Orange) */}
        <div className="bg-white p-6 rounded-2xl shadow-sm border-t-4 border-brand-orange">
          <p className="font-work-sans text-xs text-slate-500 uppercase tracking-wider">
            Feedback Positivo
          </p>
          <p className="font-poppins font-bold text-3xl text-brand-dark mt-2">
            {stats.feedback_percentage}%
          </p>
          <p className="text-brand-orange text-xs mt-2 font-semibold font-work-sans italic">
            Basado en Satellites
          </p>
        </div>

        {/* Las demás cards siguen el mismo patrón... */}
      </div>
    </div>
  );
};

export default Home;
