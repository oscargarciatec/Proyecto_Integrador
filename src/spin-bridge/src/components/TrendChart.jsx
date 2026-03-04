import React, { useEffect, useState, useCallback, useMemo } from "react";
import {
  AreaChart,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
} from "recharts";

// 1. Agregamos isDark a las props para que React detecte el cambio de tema
const CustomTooltip = ({ active, payload, label }) => {
  // Verificamos si el modo oscuro está activo en el HTML directamente para asegurar precisión
  const isDarkInHtml =
    typeof document !== "undefined" &&
    document.documentElement.classList.contains("dark");

  if (active && payload && payload.length) {
    return (
      <div
        className={`${
          isDarkInHtml
            ? "bg-slate-800 border-slate-700"
            : "bg-white border-slate-200"
        } border p-3 rounded-xl shadow-xl transition-colors duration-300`}
      >
        <p
          className={`text-[10px] font-bold uppercase mb-1 font-montserrat text-left ${
            isDarkInHtml ? "text-slate-100" : "text-slate-400"
          }`}
        >
          {label}
        </p>
        <p
          className={`text-sm font-bold font-montserrat ${
            isDarkInHtml ? "text-white" : "text-slate-900"
          }`}
        >
          <span className="text-[#00C2CB]">●</span> {payload[0].value} mensajes
        </p>
      </div>
    );
  }
  return null;
};

export const TrendChart = React.memo(({ data }) => {
  const [isDark, setIsDark] = useState(() =>
    typeof window !== "undefined"
      ? document.documentElement.classList.contains("dark")
      : false,
  );
  const [width, setWidth] = useState(0);

  useEffect(() => {
    const observer = new MutationObserver(() => {
      const isCurrentlyDark =
        document.documentElement.classList.contains("dark");
      setIsDark((prev) => (prev !== isCurrentlyDark ? isCurrentlyDark : prev));
    });

    observer.observe(document.documentElement, {
      attributes: true,
      attributeFilter: ["class"],
    });
    return () => observer.disconnect();
  }, []);

  const containerRef = useCallback((node) => {
    if (node !== null) {
      const measure = () => setWidth(node.offsetWidth);
      measure();
      window.addEventListener("resize", measure);
      return () => window.removeEventListener("resize", measure);
    }
  }, []);

  const colors = useMemo(
    () => ({
      grid: isDark ? "rgba(255, 255, 255, 0.05)" : "rgba(0, 0, 0, 0.05)",
      text: isDark ? "#ffffff" : "#94A3B8",
      line: "#00C2CB",
      fill: isDark ? "#1e293b" : "#ffffff",
    }),
    [isDark],
  );

  if (!data || data.length === 0) return null;

  return (
    <div
      ref={containerRef}
      className="w-full h-[350px] relative overflow-hidden flex justify-center"
      style={{ backgroundColor: colors.fill }}
      key={isDark ? "dark" : "light"}
    >
      {width > 0 && (
        <AreaChart
          width={width}
          height={350}
          data={data}
          margin={{ top: 10, right: 30, left: 0, bottom: 0 }}
          fill={colors.fill}
        >
          <defs>
            <linearGradient id="colorBrand" x1="0" y1="0" x2="0" y2="1">
              <stop offset="5%" stopColor={colors.line} stopOpacity={0.2} />
              <stop offset="95%" stopColor={colors.line} stopOpacity={0} />
            </linearGradient>
          </defs>

          <CartesianGrid
            strokeDasharray="4 4"
            vertical={false}
            stroke={colors.grid}
            fill={colors.fill}
          />

          <XAxis
            dataKey="fecha"
            axisLine={false}
            tickLine={false}
            tick={{ fill: colors.text, fontSize: 10, fontFamily: "Montserrat" }}
            dy={10}
          />

          <YAxis
            axisLine={false}
            tickLine={false}
            tick={{ fill: colors.text, fontSize: 10, fontFamily: "Montserrat" }}
          />

          {/* PASAMOS isDark AQUÍ PARA NOTIFICAR AL TOOLTIP */}
          <Tooltip
            content={<CustomTooltip />}
            // 2. IMPORTANTE: Esto desactiva el cache interno de renderizado de Recharts para el tooltip
            isAnimationActive={false}
            // 3. Forzamos que el cursor (la línea vertical) también cambie
            cursor={{
              stroke: isDark ? "rgba(255,255,255,0.2)" : "rgba(0,0,0,0.1)",
              strokeWidth: 1,
            }}
          />

          <Area
            type="monotone"
            dataKey="mensajes"
            stroke={colors.line}
            strokeWidth={2.5}
            fill="url(#colorBrand)"
            isAnimationActive={true}
            animationDuration={300}
          />
        </AreaChart>
      )}
    </div>
  );
});
