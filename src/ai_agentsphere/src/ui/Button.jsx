export const Button = ({ children, variant = "primary", ...props }) => {
  const variants = {
    primary: "bg-brand-orange hover:bg-brand-dark/70 text-white",
    outline:
      "border-2 border-brand-primary text-brand-primary hover:bg-brand-primary/10",
    ghost: "text-brand-purple hover:bg-brand-purple/5",
  };

  return (
    <button
      className={`px-6 py-2 rounded-full font-montserrat font-bold transition-all active:scale-95 disabled:opacity-50 ${variants[variant]}`}
      {...props}
    >
      {children}
    </button>
  );
};
