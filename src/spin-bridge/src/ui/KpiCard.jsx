import React from "react";

export const KpiCard = React.memo(
  ({ title, value, variant = "primary", fontSize = "large", children }) => {
    // Definimos las clases de color de forma EXPLÍCITA para el borde superior
    const variantColors = {
      primary: "border-t-brand-primary dark:border-t-brand-primary",
      purple: "border-t-brand-purple dark:border-t-brand-purple",
      orange: "border-t-brand-orange dark:border-t-brand-orange",
    };

    const sizeClasses = {
      small: "text-sm",
      medium: "text-md",
      large: "text-lg",
      xlarge: "text-xl",
      xxlarge: "text-2xl",
      xxxlarge: "text-4xl",
    };

    return (
      <div
        className={`
          relative p-6 rounded-2xl shadow-sm transition-all duration-300 hover:shadow-md
          /* Fondo y texto base */
          bg-white dark:bg-slate-900 
          text-brand-dark dark:text-white
          
          /* Estructura de bordes: 1px en todos lados */
          border border-slate-100 dark:border-slate-800
          
          /* El "Top Border" de color: Forzamos el ancho y el color de la variante */
          border-t-6 ${variantColors[variant]}
        `}
      >
        <p className="font-work-sans text-[11px] uppercase tracking-widest text-slate-500 dark:text-slate-400 mb-1 font-bold">
          {title}
        </p>
        <div className="flex items-end justify-between">
          <h3
            className={`font-poppins font-bold tracking-tight transition-colors ${sizeClasses[fontSize]}`}
          >
            {value}
          </h3>
          <div className="shrink-0 transition-transform duration-300 hover:scale-110">
            {children}
          </div>
        </div>
      </div>
    );
  },
  (prev, next) => {
    return (
      prev.value === next.value &&
      prev.title === next.title &&
      prev.variant === next.variant &&
      prev.children === next.children
    );
  },
);
