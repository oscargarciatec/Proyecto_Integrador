import React, { useState, useCallback, useMemo } from "react";
import {
  AreaChart,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ReferenceLine,
} from "recharts";
import { useThemeState } from "../context/ThemeContextCore";

const CustomDot = (props) => {
  const { cx, cy, payload } = props;
  if (payload.has_negative_feedback) {
    return (
      <circle
        cx={cx}
        cy={cy}
        r={4}
        stroke="none"
        fill="#EF4444"
        className="animate-pulse"
      />
    );
  }
  return null;
};

const CustomTooltip = ({ active, payload, label, isDark }) => {
  if (active && payload && payload.length) {
    return (
      <div
        className={`${
          isDark ? "bg-slate-800 border-slate-700" : "bg-white border-slate-200"
        } border p-3 rounded-xl shadow-xl transition-colors duration-300`}
      >
        <p
          className={`text-[10px] font-bold uppercase mb-1 font-montserrat text-left ${
            isDark ? "text-slate-100" : "text-slate-400"
          }`}
        >
          {label}
        </p>
        <p
          className={`text-sm font-bold font-montserrat ${
            isDark ? "text-white" : "text-slate-900"
          }`}
        >
          <span className="text-[#00C2CB]">●</span> {payload[0].value}{" "}
          {payload[0].value === 1 ? "message" : "messages"}
        </p>
        {payload[0].payload.has_negative_feedback && (
          <p className="text-[10px] font-bold text-red-500 mt-2 flex items-center gap-1 font-montserrat uppercase">
            <span className="text-sm">●</span> Negative Feedback Detected
          </p>
        )}
      </div>
    );
  }
  return null;
};

export const TrendChart = React.memo(({ data, isLoading = false }) => {
  const isDark = useThemeState();
  const [width, setWidth] = useState(0);

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

  const average = useMemo(() => {
    if (!data || data.length === 0) return 0;
    const total = data.reduce((sum, item) => sum + item.mensajes, 0);
    return Math.round(total / data.length);
  }, [data]);

  if (isLoading || !data || data.length === 0) {
    return (
      <div className="w-full h-[380px] bg-slate-50 dark:bg-slate-900/50 rounded-2xl flex items-center justify-center animate-pulse border border-slate-100 dark:border-slate-800">
        <div className="flex flex-col items-center gap-4">
          <div className="w-12 h-12 border-4 border-brand-primary/20 border-t-brand-primary rounded-full animate-spin"></div>
          <p className="text-slate-400 font-montserrat text-xs font-bold uppercase tracking-widest">
            Loading trend data...
          </p>
        </div>
      </div>
    );
  }

  return (
    <div
      ref={containerRef}
      className="w-full h-[380px] relative overflow-hidden flex flex-col items-center"
      style={{ backgroundColor: colors.fill }}
      key={isDark ? "dark" : "light"}
    >
      {width > 0 && (
        <AreaChart
          width={width}
          height={320}
          data={data}
          margin={{ top: 10, right: 30, left: 0, bottom: 0 }}
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

          <Tooltip
            content={<CustomTooltip isDark={isDark} />}
            isAnimationActive={false}
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
            dot={<CustomDot />}
          />

          <ReferenceLine
            y={average}
            stroke={isDark ? "#94A3B8" : "#64748B"}
            strokeDasharray="3 3"
            label={{
              value: `${average}`,
              position: "insideBottomRight",
              fill: isDark ? "#94A3B8" : "#64748B",
              fontSize: 10,
              fontFamily: "Montserrat",
            }}
          />
        </AreaChart>
      )}

      {/* Leyenda/Nota */}
      <div className="flex gap-6 mt-8 pb-4 pointer-events-none">
        <div className="flex items-center gap-1.5">
          <div className="w-2.5 h-2.5 rounded-full bg-red-500 animate-pulse" />
          <span
            className={`text-[10px] font-bold uppercase font-montserrat tracking-wider ${
              isDark ? "text-slate-400" : "text-slate-500"
            }`}
          >
            Negative Feedback Detected
          </span>
        </div>
        <div className="flex items-center gap-1.5">
          <div className="w-5 h-0 border-t-2 border-dashed border-slate-400" />
          <span
            className={`text-[10px] font-bold uppercase font-montserrat tracking-wider ${
              isDark ? "text-slate-400" : "text-slate-500"
            }`}
          >
            Average
          </span>
        </div>
        <div className="flex items-center gap-1.5">
          <div className="w-5 h-0 border-t-3 border-brand-primary" />
          <span
            className={`text-[10px] font-bold uppercase font-montserrat tracking-wider ${
              isDark ? "text-slate-400" : "text-slate-500"
            }`}
          >
            Messages
          </span>
        </div>
      </div>
    </div>
  );
});
