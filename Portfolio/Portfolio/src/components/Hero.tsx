import { Download, Mail } from "lucide-react";
import { Button } from "@/components/ui/button";

const Hero = () => {
  const scrollToContact = () => {
    const element = document.querySelector("#contact");
    if (element) {
      element.scrollIntoView({ behavior: "smooth", block: "start" });
    }
  };

  return (
    <section
      id="home"
      className="relative min-h-screen flex items-center justify-center overflow-hidden pt-20"
    >
      {/* Animated Background Gradient */}
      <div className="absolute inset-0 bg-gradient-to-br from-primary/10 via-background to-secondary/10">
        <div className="absolute inset-0 bg-[radial-gradient(circle_at_30%_50%,rgba(79,70,229,0.1),transparent_50%)]" />
        <div className="absolute inset-0 bg-[radial-gradient(circle_at_70%_80%,rgba(14,165,233,0.1),transparent_50%)]" />
      </div>

      {/* Floating Shapes */}
      <div className="absolute top-20 left-10 w-20 h-20 bg-primary/20 rounded-full blur-3xl animate-float" />
      <div className="absolute bottom-20 right-10 w-32 h-32 bg-secondary/20 rounded-full blur-3xl animate-float" style={{ animationDelay: "1s" }} />

      <div className="container mx-auto px-4 sm:px-6 lg:px-8 relative z-10">
        <div className="max-w-4xl mx-auto text-center">
          <div className="animate-fade-in">
            <p className="text-sm sm:text-base text-muted-foreground mb-4 tracking-wider uppercase">
              Welcome to my portfolio
            </p>
            <h1 className="text-4xl sm:text-5xl md:text-6xl lg:text-7xl font-bold mb-4 bg-gradient-to-r from-primary via-secondary to-primary bg-clip-text text-transparent animate-gradient">
              Ravi Teja Kesagani
            </h1>
            <h2 className="text-xl sm:text-2xl md:text-3xl text-foreground/80 mb-6 font-medium">
              Business Data Analyst
            </h2>
            <p className="text-base sm:text-lg md:text-xl text-muted-foreground mb-12 max-w-2xl mx-auto leading-relaxed">
              I analyze business data using SQL, Python, Excel, and Power BI to uncover patterns, risks, and revenue-impacting insights through reporting and dashboards.
            </p>

            {/* CTA Buttons */}
            <div className="flex flex-col sm:flex-row gap-4 justify-center items-center">
              <Button
                size="lg"
                className="w-full sm:w-auto bg-gradient-to-r from-primary to-secondary hover:shadow-lg hover:shadow-primary/50 transition-all duration-300 hover:scale-105"
                asChild
              >
                <a href="/Data_Analyst_Ravi_Teja_1.pdf" download>
                  <Download className="mr-2 h-5 w-5" />
                  Download Resume
                </a>
              </Button>
              <Button
                size="lg"
                variant="outline"
                className="w-full sm:w-auto border-2 hover:bg-primary/10 hover:border-primary transition-all duration-300 hover:scale-105"
                onClick={scrollToContact}
              >
                <Mail className="mr-2 h-5 w-5" />
                Contact Me
              </Button>
            </div>
          </div>

          {/* Scroll Indicator */}
          <div className="absolute bottom-8 left-1/2 transform -translate-x-1/2 animate-bounce">
            <div className="w-6 h-10 border-2 border-primary/50 rounded-full flex justify-center pt-2">
              <div className="w-1 h-2 bg-primary rounded-full" />
            </div>
          </div>
        </div>
      </div>
    </section>
  );
};

export default Hero;
