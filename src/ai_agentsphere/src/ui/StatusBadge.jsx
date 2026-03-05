export const StatusBadge = ({ status }) => {
  const config = {
    positive: {
      label: "Positive",
      classes: "bg-green-100 text-green-700 border-green-200",
    },
    negative: {
      label: "Negative",
      classes: "bg-brand-orange/10 text-brand-orange border-brand-orange/20",
    },
    none: {
      label: "No Feedback",
      classes: "bg-slate-100 text-slate-500 border-slate-200",
    },
  };

  // CORRECCIÓN: Usamos minúsculas (true/false) y manejamos el valor de la DB
  let current;
  if (status === true) {
    current = config.positive;
  } else if (status === false) {
    current = config.negative;
  } else {
    current = config.none;
  }

  return (
    <span
      className={`px-3 py-1 rounded-full text-xs font-bold border ${current.classes} font-work-sans inline-block`}
    >
      {current.label}
    </span>
  );
};
