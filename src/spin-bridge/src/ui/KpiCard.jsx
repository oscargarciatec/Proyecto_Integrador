// src/components/ui/KpiCard.jsx
export const KpiCard = ({ title, value, variant = "primary", children }) => {
  const variants = {
    primary: "border-brand-primary text-brand-dark",
    purple: "border-brand-purple text-brand-dark",
    orange: "border-brand-orange text-brand-dark",
  };

  return (
    <div
      className={`bg-white p-6 rounded-2xl shadow-sm border-t-4 ${variants[variant]} transition-all hover:shadow-md`}
    >
      <p className="font-work-sans text-xs uppercase tracking-widest text-slate-500 mb-1">
        {title}
      </p>
      <div className="flex items-end justify-between">
        <h3 className="font-poppins font-bold text-3xl tracking-tight">
          {value}
        </h3>
        <div className="shrink-0">{children}</div>
      </div>
    </div>
  );
};
