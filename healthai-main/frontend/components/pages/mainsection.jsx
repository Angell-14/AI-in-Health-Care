import { HoverEffect } from "../ui/card-hover-effect";

export function CardHoverEffectDemo() {
  return (
    (<div className="max-w-5xl mx-auto px-8 " id="mainsection">
      <HoverEffect items={projects} />
    </div>)
  );
}
export const projects = [
    {
        title: "Chatbot",
        description: "AI-powered chatbot for patient engagement and symptom assessment.",
        link: "/chat/ch"
      },
      {
        title: "Personalize Treatment Plans",
        description: "Customized treatment recommendations based on patient data.",
        link: "/chat/p"
      },
      {
        title: "Generate Data",
        description: "Customized and synthesized data",
        link: "/chat/d"
      }
];
