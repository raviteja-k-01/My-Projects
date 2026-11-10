import { useEffect } from "react";
import Navbar from "@/components/Navbar";
import Hero from "@/components/Hero";
import About from "@/components/About";
import Skills from "@/components/Skills";
import Projects from "@/components/Projects";
import Experience from "@/components/Experience";
import Contact from "@/components/Contact";
import { Separator } from "@/components/ui/separator";

const Index = () => {
  useEffect(() => {
    // Add loading animation
    const timer = setTimeout(() => {
      document.body.classList.add("loaded");
    }, 100);

    return () => clearTimeout(timer);
  }, []);

  return (
    <div className="min-h-screen bg-background">
      <Navbar />
      <main>
        <Hero />
        <Separator className="my-0" />
        <About />
        <Separator className="my-0" />
        <Skills />
        <Separator className="my-0" />
        <Projects />
        <Separator className="my-0" />
        <Experience />
        <Separator className="my-0" />
        <Contact />
      </main>
    </div>
  );
};

export default Index;
