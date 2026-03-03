export const ChartContainer = ({ title, subtitle, children }) => {
  return (
    <div className="bg-white p-6 rounded-2xl shadow-sm border border-slate-100 flex flex-col w-full overflow-hidden">
      <div className="mb-6">
        <h3 className="font-montserrat font-bold text-brand-dark text-lg">
          {title}
        </h3>
        {subtitle && (
          <p className="font-work-sans text-sm text-slate-500 mt-1">
            {subtitle}
          </p>
        )}
      </div>

      {/* Contenedor rígido para evitar el error de altura negativa */}
      <div className="relative block w-full" style={{ minHeight: "350px" }}>
        {children}
      </div>
    </div>
  );
};
