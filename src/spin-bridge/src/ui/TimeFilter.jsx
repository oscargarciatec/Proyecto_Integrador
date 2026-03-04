export const TimeFilter = ({ selected, onChange }) => {
  const options = [
    { label: "7 Días", value: 7 },
    { label: "30 Días", value: 30 },
  ];

  return (
    <div className="flex bg-slate-100 p-1 rounded-xl w-fit border border-slate-200 dark:bg-slate-800 dark:border-slate-700">
      {options.map((opt) => (
        <button
          key={opt.value}
          onClick={() => onChange(opt.value)}
          className={`px-4 py-1.5 rounded-lg text-sm font-montserrat font-bold transition-all ${
            selected === opt.value
              ? "bg-white text-brand-purple shadow-sm ring-1 ring-slate-200"
              : "text-slate-500 hover:text-brand-primary"
          }`}
        >
          {opt.label}
        </button>
      ))}
    </div>
  );
};
