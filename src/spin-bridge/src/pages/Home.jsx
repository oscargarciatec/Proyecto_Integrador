import { useApi } from "../hooks/useApi";
import { TimeFilter } from "../ui/TimeFilter";
import { Users, Crown, MessageCircle, MessageCircleHeart } from "lucide-react";
import { KpiCard } from "../ui/KpiCard";
import { ChartContainer } from "../ui/ChartContainer";
import { TrendChart } from "../components/TrendChart";

const Home = ({ days, setDays }) => {
  const { data: stats } = useApi(`/api/dashboard/stats?days=${days}`);
  const { data: trend } = useApi(`/api/dashboard/trend?days=${days}`);

  return (
    <div className="p-8 space-y-8 bg-[#F8FAFC] dark:bg-slate-800">
      <header>
        <div>
          <h2 className="text-4xl font-montserrat font-bold text-brand-dark dark:text-slate-300">
            Dashboard Ejecutivo
          </h2>
          <p className="text-slate-500 font-work-sans">
            Métricas globales de{" "}
            <span className="text-brand-primary font-semibold">Compass</span>
          </p>
        </div>
        <TimeFilter selected={days} onChange={setDays} />
      </header>

      {/* Grid de KPIs Reusables */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-5 gap-6">
        <KpiCard
          title="Usuarios Históricos"
          value={stats?.total_users || 0}
          variant="primary"
          fontSize="xxxlarge"
        >
          <Users
            className="text-brand-primary/40 dark:text-brand-primary"
            size={24}
          />
        </KpiCard>

        <KpiCard
          title="Usuarios Activos en el periodo"
          value={stats?.active_users}
          variant="purple"
          fontSize="xxxlarge"
        >
          <Users
            className="text-brand-purple/40 dark:text-brand-white"
            size={24}
          />
        </KpiCard>

        <KpiCard
          title="Conversaciones"
          value={stats?.total_conversations}
          variant="purple"
          fontSize="xxxlarge"
        >
          <MessageCircle
            className="text-brand-primary/40 dark:text-brand-primary"
            size={24}
          />
        </KpiCard>

        <KpiCard
          title="% Feedback Positivo"
          value={stats?.feedback_percentage + "%"}
          variant="purple"
          fontSize="xxxlarge"
        >
          <MessageCircleHeart
            className="text-brand-primary/40 dark:text-brand-primary"
            size={24}
          />
        </KpiCard>

        <KpiCard
          title="Top User"
          value={stats?.top_user}
          variant="orange"
          fontSize="xlarge"
        >
          <Crown
            className="text-brand-orange/40 dark:text-brand-orange"
            size={24}
          />
        </KpiCard>
      </div>
      {/* Gráfica Reusable */}
      <div className="grid grid-cols-1 gap-8 w-full overflow-hidden">
        <ChartContainer
          title={`Tendencia de los últimos ${days} días`}
          subtitle="Volumen de mensajes detectados en Slack en el periodo"
        >
          <TrendChart data={trend} />
        </ChartContainer>
      </div>
    </div>
  );
};

export default Home;
