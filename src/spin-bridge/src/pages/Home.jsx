import { useApi } from "../hooks/useApi";
import { TimeFilter } from "../ui/TimeFilter";
import { Users, Crown, MessageCircle, MessageCircleHeart } from "lucide-react";
import { KpiCard } from "../ui/KpiCard";
import { ChartContainer } from "../ui/ChartContainer";
import { TrendChart } from "../components/TrendChart";
import { theme } from "../styles/theme";

const Home = ({ days, setDays }) => {
  const { data: stats } = useApi(`/api/dashboard/stats?days=${days}`);
  const { data: trend } = useApi(`/api/dashboard/trend?days=${days}`);

  return (
    <div className="p-8 space-y-8 bg-[#F8FAFC]">
      <header>
        <div>
          <h2 className="text-4xl font-montserrat font-bold text-brand-dark">
            Dashboard Ejecutivo
          </h2>
          <p className="text-slate-500 font-work-sans">
            Métricas globales de{" "}
            <span className="text-brand-primary font-semibold">
              Spin Compass
            </span>
          </p>
        </div>
        <TimeFilter selected={days} onChange={setDays} />
      </header>

      {/* Grid de KPIs Reusables */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        <KpiCard
          title="Usuarios Históricos"
          value={stats?.total_users || 0}
          variant="primary"
        >
          <Users className="text-brand-primary/40" size={24} />
        </KpiCard>
        <KpiCard
          title="Conversaciones"
          value={stats?.total_conversations}
          variant="primary"
        >
          <MessageCircle className="text-brand-primary/40" size={24} />
        </KpiCard>

        <KpiCard
          title="% Feedback Positivo"
          value={stats?.feedback_percentage + "%"}
          variant="primary"
        >
          <MessageCircleHeart className="text-brand-primary/40" size={24} />
        </KpiCard>

        <KpiCard
          title="Usuarios Activos en el periodo"
          value={stats?.active_users}
          variant="purple"
        >
          <Users className="text-brand-purple/40" size={24} />
        </KpiCard>

        <KpiCard title="Top User" value={stats?.top_user} variant="orange">
          <Crown className="text-brand-orange/40" size={24} />
        </KpiCard>
      </div>
      {/* Gráfica Reusable */}
      <div className="grid grid-cols-1 gap-8 w-full overflow-hidden">
        <ChartContainer
          title="Tendencia Semanal"
          subtitle="Volumen de mensajes detectados en Slack"
        >
          <TrendChart data={trend} />
        </ChartContainer>
      </div>
    </div>
  );
};

export default Home;
