import { useState } from "react";
import { useApi } from "../hooks/useApi";
import { useModal } from "../hooks/useModal";
import { DataTable } from "../ui/DataTable";
import { Modal } from "../ui/Modal";
import { LoadingState } from "../ui/LoadingState";
import { StatusBadge } from "../ui/StatusBadge";

const Conversations = () => {
  const [filterEmail, setFilterEmail] = useState("");
  const { isOpen, content: selectedConv, openModal, closeModal } = useModal();

  // Hook para la tabla de negativos
  const endpoint = `/api/conversations/negative${filterEmail ? `?email=${filterEmail.trim()}` : ""}`;

  const { data: negChats, loading } = useApi(endpoint);

  // Hook para el detalle (se dispara cuando seleccionamos una fila)
  const { data: thread } = useApi(
    selectedConv
      ? `/api/conversations/negative/${selectedConv.kh_conversation}`
      : null,
  );

  const columns = [
    { label: "Usuario", key: "user" },
    { label: "Fecha", key: "date" },
    {
      label: "Estado",
      key: "status",
      render: () => <StatusBadge status={false} />,
    },
    {
      label: "Mensaje con Error",
      key: "snippet",
      render: (val) => <span className="italic">"{val}"</span>,
    },
    {
      label: "Comentario del Usuario",
      key: "comment",
      render: (val) => (
        <span className="text-brand-orange font-work-sans font-medium italic">
          {val !== "Sin comentario" ? `"${val}"` : val}
        </span>
      ),
    },
  ];

  if (loading) return <LoadingState />;

  return (
    <div className="p-8 space-y-6">
      <header className="flex justify-between items-end">
        <div>
          <h2 className="text-3xl font-montserrat font-bold text-brand-dark">
            Auditoría de Feedback
          </h2>
          <p className="text-brand-orange font-work-sans font-bold text-sm">
            Experiencias negativas
          </p>
        </div>
        <input
          placeholder="Filtrar por correo..."
          className="p-2 border rounded-xl font-work-sans text-sm outline-none focus:ring-2 focus:ring-brand-primary"
          onChange={(e) => setFilterEmail(e.target.value)}
        />
      </header>

      <DataTable
        columns={columns}
        data={negChats || []}
        onRowClick={openModal}
      />

      {/* Modal para ver el hilo completo */}
      <Modal
        isOpen={isOpen}
        onClose={closeModal}
        title={`Historial: ${selectedConv?.user}`}
      >
        <div className="space-y-4">
          {thread?.map((msg, i) => (
            <div
              key={i}
              className={`flex ${msg.type === "bot_response" ? "justify-start" : "justify-end"}`}
            >
              <div
                className={`max-w-[80%] p-4 rounded-2xl font-work-sans text-sm ${
                  msg.type === "bot_response"
                    ? "bg-slate-100 text-brand-dark rounded-tl-none"
                    : "bg-brand-primary text-white rounded-tr-none"
                }`}
              >
                <p>{msg.content}</p>
                <p className="text-[10px] mt-2 opacity-50">{msg.timestamp}</p>
                {msg.feedback === false && (
                  <div className="mt-2 text-[10px] bg-red-500 text-white px-2 py-1 rounded inline-block font-bold">
                    ESTE MENSAJE RECIBIÓ FEEDBACK NEGATIVO
                  </div>
                )}
              </div>
            </div>
          ))}
          {thread?.find((m) => m.comment) && (
            <div className="mt-6 p-4 bg-brand-orange/5 border border-brand-orange/20 rounded-xl">
              <h4 className="text-xs font-bold text-brand-orange uppercase tracking-wider mb-2 font-montserrat">
                Feedback del Usuario:
              </h4>
              <p className="text-brand-dark font-work-sans italic text-sm">
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
