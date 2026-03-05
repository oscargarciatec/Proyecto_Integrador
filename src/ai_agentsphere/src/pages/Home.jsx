import React, { useMemo } from "react";
import { useApi } from "../hooks/useApi";
import { TimeFilter } from "../ui/TimeFilter";
import {
  Users,
  Crown,
  MessageCircle,
  MessageCircleHeart,
  CircleGauge,
} from "lucide-react";
import { KpiCard } from "../ui/KpiCard";
import { ChartContainer } from "../ui/ChartContainer";
import { TrendChart } from "../components/TrendChart";

const Home = React.memo(({ days, setDays }) => {
  const { data: dashboardData } = useApi(
    `/api/dashboard/combined?days=${days}`,
  );

  const stats = useMemo(() => dashboardData?.stats, [dashboardData]);
  const trend = useMemo(() => dashboardData?.trend, [dashboardData]);

  // Memoizamos los iconos para que no cambien la referencia en cada render
  // y así el React.memo del KpiCard sea efectivo.
  const icons = useMemo(
    () => ({
      users: (
        <Users
          className="text-brand-primary/40 dark:text-brand-primary"
          size={24}
        />
      ),
      activeUsers: (
        <Users
          className="text-brand-purple/40 dark:text-brand-white"
          size={24}
        />
      ),
      conversations: (
        <MessageCircle
          className="text-brand-primary/40 dark:text-brand-primary"
          size={24}
        />
      ),
      feedback: (
        <MessageCircleHeart
          className="text-brand-primary/40 dark:text-brand-primary"
          size={24}
        />
      ),
      crown: (
        <Crown
          className="text-brand-orange/40 dark:text-brand-orange"
          size={24}
        />
      ),
    }),
    [],
  );

  return (
    <div className="p-8 space-y-8 bg-[#F8FAFC] dark:bg-slate-800">
      <header>
        <div>
          <h2 className="text-4xl font-montserrat font-bold text-brand-dark dark:text-slate-300">
            <CircleGauge size={34} className="inline" /> Metrics Center
          </h2>
          <p className="text-slate-500 dark:text-slate-400 font-work-sans h-10">
            An AIOps metrics center for{" "}
            <span className="text-brand-primary font-semibold">Compass</span>
          </p>
        </div>
        <TimeFilter selected={days} onChange={setDays} />
      </header>

      {/* Grid de KPIs Reusables */}
      <div className="grid grid-cols-1 md:grid-cols-4 lg:grid-cols-4 xl:grid-cols-5 gap-6">
        <KpiCard
          title="Historical Users"
          value={stats?.total_users || 0}
          variant="primary"
          fontSize="xxxlarge"
        >
          {icons.users}
        </KpiCard>

        <KpiCard
          title="Active Users"
          value={stats?.active_users}
          variant="purple"
          fontSize="xxxlarge"
        >
          {icons.activeUsers}
        </KpiCard>

        <KpiCard
          title="Conversations"
          value={stats?.total_conversations}
          variant="purple"
          fontSize="xxxlarge"
        >
          {icons.conversations}
        </KpiCard>

        <KpiCard
          title="Positive Feedback %"
          value={stats?.feedback_percentage + "%"}
          variant="purple"
          fontSize="xxxlarge"
        >
          {icons.feedback}
        </KpiCard>

        <KpiCard
          title="Top User"
          value={stats?.top_user}
          variant="orange"
          fontSize="xlarge"
        >
          {icons.crown}
        </KpiCard>
      </div>
      {/* Gráfica Reusable */}
      <div className="grid grid-cols-1 gap-8 w-full overflow-hidden">
        <ChartContainer
          title={`Trend of the last ${days} days`}
          subtitle="Volume of messages detected in Slack in the selected period"
        >
          <TrendChart data={trend} />
        </ChartContainer>
      </div>
    </div>
  );
});

export default Home;
