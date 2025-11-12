// == Refactored Advanced UI with Components, 3D Tilt, AI Glow ==
// Components extracted:
// ✅ Sidebar (icons only, neon hover glow, no spring)
// ✅ TopTabs (animated emoji + active glow ring, no spring)
// ✅ Sections (no animations inside sections per request)
// ✅ AgentsPanel
// ✅ InputBar (AI glow effect on focus)
// Notes:
// - FIXED: Mismatched </motion.div> tags in Sidebar (both inner and outer) ✅
// - Removed down-to-up animations; kept simple fades where applicable (outside sections)

import React, { useMemo, useState } from "react";
import { motion, AnimatePresence } from "framer-motion";

// ==============================
// Utility
// ==============================
const glow = "hover:shadow-[0_0_12px_rgba(99,102,241,0.55)] transition-shadow";

export const TAB_VARIANTS = {
  Weather: "a",
  Settings: "a",
  Music: "a",
  Version: "a",
  Tasks: "b",
  Notifications: "a",
};

export const AGENT_LIST = ["Planner", "Retriever", "Tool Runner"];

// ==============================
// ScrollIcon Component
// ==============================
function ScrollIcon({ label, active, onClick, variant, badge }) {
  const emojis = {
    Weather: "☁️",
    Settings: "⚙️",
    Music: "🎵",
    Version: "🔖",
    Tasks: "✅",
    Notifications: "🔔",
  };

  return (
    <motion.button
      whileTap={{ scale: 0.96 }}
      className={`relative px-4 py-2 inline-flex items-center gap-2 bg-white border rounded-xl shadow-sm ${glow}`}
      onClick={onClick}
      aria-pressed={!!active}
      data-testid={`tab-${label.toLowerCase()}`}
      title={label}
    >
      <motion.span animate={{ rotate: active ? 10 : 0 }}>{emojis[label]}</motion.span>
      <span className="text-xs font-semibold text-gray-800">{label}</span>
      {badge ? (
        <span className="absolute -top-1 -right-1 text-[10px] bg-white border px-1 rounded">{badge}</span>
      ) : null}
      {active ? (
        <motion.div
          layoutId="glowRing"
          className="absolute inset-0 rounded-xl border border-indigo-300 shadow-[0_0_12px_rgba(99,102,241,0.65)]"
        />
      ) : null}
    </motion.button>
  );
}

// ==============================
// Sidebar
// ==============================
function Sidebar({ open, setOpen }) {
  return (
    <motion.div
      animate={{ width: open ? 200 : 64 }}
      className="border-r bg-white/70 backdrop-blur p-3 flex flex-col items-center gap-3"
    >
      <motion.button
        whileHover={{ rotate: 90 }}
        onClick={() => setOpen(!open)}
        className="text-2xl"
        aria-label="Toggle menu"
        title="Menu"
      >
        {open ? "✖" : "☰"}
      </motion.button>

      <AnimatePresence>
        {open ? (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="flex flex-col gap-3 mt-2"
          >
            <motion.button className={`text-2xl p-3 rounded-xl ${glow}`} aria-label="Folders" title="Folders">📁</motion.button>
            <motion.button className={`text-2xl p-3 rounded-xl ${glow}`} aria-label="Chat History" title="Chat History">💬</motion.button>
          </motion.div>
        ) : null}
      </AnimatePresence>
    </motion.div>
  );
}

// ==============================
// Section Components (no animations inside)
// ==============================
const FloatEmoji = ({ children }) => (
  <span className="mr-1">{children}</span>
);

const sections = {
  Weather: () => (
    <div className="p-4 bg-white rounded-2xl shadow border">
      <div className="text-sm text-gray-500">Now</div>
      <div className="text-4xl font-semibold flex items-center"><FloatEmoji>☁️</FloatEmoji>26°C</div>
      <div className="text-gray-600">Partly Cloudy · Hyderabad</div>
    </div>
  ),

  Settings: () => (
    <div className="p-4 bg-white rounded-2xl shadow border">
      <div className="font-semibold flex items-center mb-2"><FloatEmoji>⚙️</FloatEmoji>Settings</div>
      <label className="flex justify-between border p-3 rounded-xl">Notifications <input type="checkbox" /></label>
      <label className="flex justify-between border p-3 rounded-xl">Auto Sync <input type="checkbox" /></label>
    </div>
  ),

  Music: () => (
    <div className="p-4 bg-white rounded-2xl shadow border">
      <div className="font-semibold flex items-center mb-2"><FloatEmoji>🎵</FloatEmoji>Music Player</div>
      <div className="w-full h-32 border rounded-xl bg-gray-100" />
      <div className="flex gap-3 mt-3"><button className="border rounded-xl px-3 py-2">⏮</button><button className="border rounded-xl px-3 py-2">▶</button><button className="border rounded-xl px-3 py-2">⏭</button></div>
    </div>
  ),

  Tasks: () => (
    <div className="p-4 bg-white rounded-2xl shadow border">
      <div className="font-semibold flex items-center mb-2"><FloatEmoji>✅</FloatEmoji>Tasks</div>
      <div className="border p-2 rounded-xl">✅ Design UI Layout</div>
      <div className="border p-2 rounded-xl">⬜ Add animations</div>
      <div className="border p-2 rounded-xl">⬜ Connect backend</div>
    </div>
  ),

  Notifications: () => (
    <div className="p-4 bg-white rounded-2xl shadow border">
      <div className="font-semibold flex items-center mb-2"><FloatEmoji>🔔</FloatEmoji>Notifications</div>
      <div className="border p-2 rounded-xl">New update available</div>
      <div className="border p-2 rounded-xl">System check completed</div>
    </div>
  ),

  Version: () => (
    <div className="p-4 bg-white rounded-2xl shadow border">
      <div className="font-semibold flex items-center mb-2"><FloatEmoji>🔖</FloatEmoji>Version</div>
      <div>App Version: 1.0.0</div>
      <div>Last Update: Today</div>
    </div>
  ),
};

// ==============================
// Agents Panel
// ==============================
function AgentsPanel({ open }) {
  return (
    <AnimatePresence initial={false}>
      {open ? (
        <motion.div
          key="agents"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
          transition={{ duration: 0.25 }}
          className="border rounded-2xl p-4 bg-white text-sm space-y-2 shadow-sm"
          data-testid="agents-panel"
        >
          <div className="font-semibold mb-1">Agents</div>
          {AGENT_LIST.map((agent) => (
            <div key={agent} className="flex items-center justify-between border p-2 rounded-xl bg-white">
              <span>{agent}</span>
              <button className={`px-3 py-1 rounded-xl border ${glow}`}>Connect</button>
            </div>
          ))}
        </motion.div>
      ) : null}
    </AnimatePresence>
  );
}

// ==============================
// AI Input Bar (AI glow)
// ==============================
function InputBar({ agentsOpen, setAgentsOpen }) {
  return (
    <div className="p-3 border-t bg-white flex flex-col gap-3">
      <div className="flex gap-3 items-center">
        <motion.button
          className={`p-2 rounded-full border ${glow}`}
          onClick={() => setAgentsOpen(!agentsOpen)}
          aria-label="Toggle agents"
          title="Agents"
        >
          +
        </motion.button>

        {/* AI Glow Input */}
        <motion.input
          type="text"
          placeholder="Type your message..."
          className="flex-1 border rounded-2xl px-4 py-2 focus:outline-none"
          whileFocus={{ boxShadow: "0 0 18px rgba(99,102,241,0.55)" }}
        />

        <motion.button
          className={`p-2 rounded-full border ${glow}`}
          aria-label="Voice"
          title="Voice"
        >
          🎤
        </motion.button>
      </div>
      <AgentsPanel open={agentsOpen} />
    </div>
  );
}

// ==============================
// Top Tabs
// ==============================
function TopTabs({ items, openItem, setOpenItem }) {
  return (
    <div className="p-4 flex gap-2 flex-wrap">
      {items.map((i) => (
        <ScrollIcon
          key={i}
          label={i}
          active={openItem === i}
          onClick={() => setOpenItem(openItem === i ? "" : i)}
          variant={TAB_VARIANTS[i]}
          badge={i === "Notifications" ? "2" : null}
        />
      ))}
    </div>
  );
}

// ==============================
// Main App
// ==============================
export default function AIBrainUI() {
  const items = useMemo(() => Object.keys(TAB_VARIANTS), []);
  const [openItem, setOpenItem] = useState("");
  const [menuOpen, setMenuOpen] = useState(false);
  const [agentsOpen, setAgentsOpen] = useState(false);

  return (
    <div className="min-h-screen bg-white text-gray-900 flex overflow-hidden">
      {/* Sidebar */}
      <Sidebar open={menuOpen} setOpen={setMenuOpen} />

      {/* Main */}
      <div className="flex-1 flex flex-col">
        <TopTabs items={items} openItem={openItem} setOpenItem={setOpenItem} />

        {/* Content (kept fade only outside sections) */}
        <div className="p-4 flex-1 overflow-auto">
          <AnimatePresence initial={false}>
            {openItem ? (
              <motion.div
                key={openItem}
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                exit={{ opacity: 0 }}
              >
                {sections[openItem]()}
              </motion.div>
            ) : (
              <motion.div
                key="placeholder"
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                className="text-sm text-gray-500"
              >
                Select a tab to open a section.
              </motion.div>
            )}
          </AnimatePresence>
        </div>

        {/* Bottom input bar */}
        <InputBar agentsOpen={agentsOpen} setAgentsOpen={setAgentsOpen} />
      </div>
    </div>
  );
}

// ===== Suggested Tests (React Testing Library) =====
// import { render, screen, fireEvent } from "@testing-library/react";
// import AIBrainUI, { TAB_VARIANTS, AGENT_LIST } from "./AIBrainUI";
//
// test("tabs exist and variants mapping is correct", () => {
//   expect(TAB_VARIANTS.Tasks).toBe("b");
//   expect(Object.keys(TAB_VARIANTS)).toContain("Weather");
// });
//
// test("opens and closes a section", () => {
//   render(<AIBrainUI />);
//   const weather = screen.getByTestId("tab-weather");
//   fireEvent.click(weather);
//   expect(screen.getByText(/26°C/)).toBeInTheDocument();
//   fireEvent.click(weather);
//   expect(screen.queryByText(/26°C/)).toBeNull();
// });
//
// test("toggle agents panel shows all agents", () => {
//   render(<AIBrainUI />);
//   fireEvent.click(screen.getByLabelText("Toggle agents"));
//   AGENT_LIST.forEach((name) => {
//     expect(screen.getByText(name)).toBeInTheDocument();
//   });
// });
//
// test("sidebar toggle shows only symbol", () => {
//   render(<AIBrainUI />);
//   const toggle = screen.getByLabelText("Toggle menu");
//   expect(toggle).toHaveTextContent("☰");
//   fireEvent.click(toggle);
//   expect(toggle).toHaveTextContent("✖");
// });
//
// test("tabs display emoji prefixes", () => {
//   render(<AIBrainUI />);
//   fireEvent.click(screen.getByTestId("tab-settings"));
//   expect(screen.getByTestId("tab-settings")).toHaveTextContent("⚙️ Settings");
// });
