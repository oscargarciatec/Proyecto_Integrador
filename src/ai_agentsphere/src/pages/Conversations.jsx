import { useState, useMemo } from "react"; // Agregamos useMemo por eficiencia
import { useApi } from "../hooks/useApi";
import { useModal } from "../hooks/useModal";
import { DataTable } from "../ui/DataTable";
import { Modal } from "../ui/Modal";
import { LoadingState } from "../ui/LoadingState";
import { StatusBadge } from "../ui/StatusBadge";
import { TimeFilter } from "../ui/TimeFilter";
import { ThumbsDown } from "lucide-react";

const Conversations = ({ days, setDays }) => {
  const [filterEmail, setFilterEmail] = useState("");
  const { isOpen, content: selectedConv, openModal, closeModal } = useModal();

  // 1. Construcción segura del endpoint
  const endpoint = useMemo(() => {
    const params = new URLSearchParams();
    params.append("days", days || 7);
    if (filterEmail?.trim()) params.append("email", filterEmail.trim());
    return `/api/conversations/negative?${params.toString()}`;
  }, [days, filterEmail]);

  const { data: negChats, loading } = useApi(endpoint);

  const { data: thread } = useApi(
    selectedConv?.kh_conversation
      ? `/api/conversations/negative/${selectedConv.kh_conversation}`
      : null,
  );

  // 2. Función de fecha blindada contra null/undefined
  const formatDateToCDMX = (dateInput) => {
    if (!dateInput) return "---";

    try {
      // 1. Separamos el string: "04/03/26 00:05" -> ["04", "03", "26", "00", "05"]
      const parts = String(dateInput).split(/[/ :]/);
      if (parts.length < 5) return String(dateInput);

      let day = parseInt(parts[0]);
      let month = parseInt(parts[1]);
      let year = parseInt(parts[2]) + 2000; // Convertimos "26" en 2026
      let hour = parseInt(parts[3]);
      let minute = parseInt(parts[4]);

      // 2. Resta manual de 6 horas
      hour -= 6;

      // 3. Lógica de ajuste de día si la hora es negativa
      if (hour < 0) {
        hour += 24;
        day -= 1;

        if (day === 0) {
          month -= 1;
          // Ajuste simple para Marzo -> Febrero (2026 no es bisiesto)
          if (month === 2) day = 28;
          else if ([1, 3, 5, 7, 8, 10, 12].includes(month)) day = 31;
          else day = 30;

          if (month === 0) {
            month = 12;
            year -= 1;
          }
        }
      }

      // 4. Reconstruimos el string final
      const fDay = String(day).padStart(2, "0");
      const fMonth = String(month).padStart(2, "0");
      const fHour = String(hour).padStart(2, "0");
      const fMin = String(minute).padStart(2, "0");

      return `${fDay}/${fMonth}/${year % 100} ${fHour}:${fMin}`;
    } catch (err) {
      return String(dateInput);
    }
  };

  const columns = [
    { label: "User", key: "user" },
    { label: "Email", key: "email" },
    {
      label: "Date",
      key: "date",
      render: (val) => {
        // Esto nos dirá en pantalla qué está recibiendo la función
        return formatDateToCDMX(val);
      },
    },
    {
      label: "Status",
      key: "status",
      render: () => <StatusBadge status={false} />,
    },
    {
      label: "Response with negative feedback",
      key: "snippet",
      render: (val) => <span className="italic">"{val || "No response"}"</span>,
    },
    {
      label: "User Feedback",
      key: "comment",
      render: (val) => (
        <span className="text-brand-orange font-work-sans font-medium italic">
          {val && val !== "No comment" ? `"${val}"` : "No comment"}
        </span>
      ),
    },
  ];

  if (loading) return <LoadingState />;

  return (
    <div className="p-8 space-y-4">
      <header className="flex justify-between items-end bg-white dark:bg-brand-primary/10 p-6 rounded-2xl shadow-sm border border-slate-100 dark:border-slate-700">
        <div className="space-y-2">
          <div>
            <h2 className="text-3xl font-montserrat font-bold text-brand-dark dark:text-slate-300 tracking-tight">
              <ThumbsDown size={20} className="inline mr-2" />
              Conversations Analysis
            </h2>
            <p className="text-brand-orange font-work-sans font-bold text-sm">
              A place where you can analyze responses with negative feedback and
              the associated user comments.
            </p>
          </div>
          <TimeFilter selected={days} onChange={setDays} />
        </div>

        <div className="flex items-center gap-4">
          <input
            id="email-filter"
            name="email"
            autoComplete="email"
            placeholder="Filter by email..."
            className="p-3 border border-slate-200 dark:border-slate-700 dark:text-slate-100 rounded-xl font-work-sans text-sm outline-none focus:ring-2 focus:ring-brand-primary w-72 bg-slate-50 dark:bg-slate-800/40  transition-all"
            onChange={(e) => setFilterEmail(e.target.value)}
          />
        </div>
      </header>

      <DataTable
        columns={columns}
        data={Array.isArray(negChats) ? negChats : []}
        onRowClick={openModal}
      />

      <Modal
        isOpen={isOpen}
        onClose={closeModal}
        title={selectedConv ? `History: ${selectedConv.user}` : "Loading..."}
      >
        <div className="space-y-4 max-h-[60vh] overflow-y-auto pr-2 custom-scrollbar">
          {thread?.map((msg, i) => (
            <div
              key={i}
              className={`flex ${msg.type === "bot_response" ? "justify-start" : "justify-end"}`}
            >
              <div
                className={`max-w-[85%] p-4 rounded-2xl font-work-sans text-sm shadow-sm ${
                  msg.type === "bot_response"
                    ? "bg-slate-100 text-brand-dark rounded-tl-none border border-slate-200"
                    : "bg-brand-primary text-white rounded-tr-none"
                }`}
              >
                <p className="leading-relaxed">{msg.content}</p>
                <p className="text-[10px] mt-2 opacity-60 font-mono">
                  {formatDateToCDMX(msg.timestamp)}
                </p>
                {msg.feedback === false && (
                  <div className="mt-3 text-[9px] bg-red-500 text-white px-2 py-1 rounded-md inline-block font-bold uppercase tracking-wider">
                    Negative Feedback
                  </div>
                )}
              </div>
            </div>
          ))}

          {thread?.find((m) => m.comment) && (
            <div className="mt-6 p-4 bg-brand-orange/10 border-l-4 border-brand-orange rounded-r-xl">
              <h4 className="text-[10px] font-bold text-brand-orange uppercase tracking-widest mb-1 font-montserrat">
                User Feedback
              </h4>
              <p className="text-brand-dark dark:text-slate-100 font-work-sans italic text-sm leading-snug">
                "{thread.find((m) => m.comment).comment}"
              </p>
            </div>
          )}
        </div>
      </Modal>
    </div>
  );
};

export default Conversations;
