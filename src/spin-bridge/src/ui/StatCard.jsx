// src/components/ui/StatCard.jsx
export const StatCard = ({ title, value, footer, variant = "default" }) => {
  const styles = {
    default: "bg-white border-t-4 border-brand-primary",
    accent: "bg-brand-primary text-white shadow-brand-primary/20",
    dark: "bg-brand-dark text-white",
  };

  return (
    <div className={`p-6 rounded-2xl shadow-sm ${styles[variant]}`}>
      <p className="font-work-sans text-xs uppercase tracking-wider opacity-80">
        {title}
      </p>
      <p className="font-poppins font-bold text-3xl mt-2">{value}</p>
      {footer && <div className="mt-4 text-xs font-semibold">{footer}</div>}
    </div>
  );
};
