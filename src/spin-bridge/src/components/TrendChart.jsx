import {
  AreaChart,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
} from "recharts";

export const TrendChart = ({ data }) => {
  // 1. Verificación inmediata de datos
  if (!data || data.length === 0) {
    return (
      <div className="flex items-center justify-center h-[350px] text-slate-400 font-work-sans italic">
        Esperando datos de la Control Tower...
      </div>
    );
  }

  return (
    <div className="w-full" style={{ height: "350px" }}>
      {/* CLAVE MAESTRA: Usamos height={350} (número) en lugar de "100%".
        Esto obliga a Recharts a renderizar con una medida fija 
        mientras el CSS del Grid termina de acomodarse.
      */}
      <ResponsiveContainer width="100%" height={350}>
        <AreaChart
          data={data}
          margin={{ top: 10, right: 10, left: -20, bottom: 0 }}
        >
          <defs>
            <linearGradient id="colorPrimary" x1="0" y1="0" x2="0" y2="1">
              <stop offset="5%" stopColor="#61B5CC" stopOpacity={0.3} />
              <stop offset="95%" stopColor="#61B5CC" stopOpacity={0} />
            </linearGradient>
          </defs>
          <CartesianGrid
            strokeDasharray="3 3"
            vertical={false}
            stroke="#f1f5f9"
          />
          <XAxis
            dataKey="fecha"
            axisLine={false}
            tickLine={false}
            tick={{ fill: "#64748b", fontSize: 10, fontFamily: "Poppins" }}
          />
          <YAxis
            axisLine={false}
            tickLine={false}
            tick={{ fill: "#64748b", fontSize: 10, fontFamily: "Poppins" }}
          />
          <Tooltip
            contentStyle={{
              borderRadius: "12px",
              border: "none",
              boxShadow: "0 4px 12px rgba(0,0,0,0.1)",
              fontFamily: "Montserrat",
            }}
          />
          <Area
            type="monotone"
            dataKey="mensajes"
            stroke="#61B5CC"
            strokeWidth={3}
            fill="url(#colorPrimary)"
            isAnimationActive={false} // Desactivar animación ayuda a la estabilidad inicial
          />
        </AreaChart>
      </ResponsiveContainer>
    </div>
  );
};
