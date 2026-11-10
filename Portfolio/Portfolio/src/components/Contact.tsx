import { useState } from "react";
import { Mail, Linkedin, Github, Phone } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import { useToast } from "@/hooks/use-toast";

const Contact = () => {
  const { toast } = useToast();
  const [formData, setFormData] = useState({
    name: "",
    email: "",
    message: "",
  });

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    toast({
      title: "Message Sent!",
      description: "Thank you for reaching out. I'll get back to you soon.",
    });
    setFormData({ name: "", email: "", message: "" });
  };

  const contactLinks = [
    {
      icon: Linkedin,
      label: "LinkedIn",
      href: "https://www.linkedin.com/in/ravitejakesagani1",
      color: "text-[#0077B5] hover:bg-[#0077B5]/10",
    },
    {
      icon: Github,
      label: "GitHub",
      href: "https://github.com/raviteja-k-01",
      color: "text-foreground hover:bg-accent",
    },
    {
      icon: Phone,
      label: "9347167572",
      href: "tel:+919347167572",
      color: "text-primary hover:bg-primary/10",
    },
    {
      icon: Mail,
      label: "Email",
      href: "mailto:raviteja.inboxx@gmail.com",
      color: "text-secondary hover:bg-secondary/10",
    },
  ];

  return (
    <section id="contact" className="py-20 relative overflow-hidden bg-gradient-to-b from-background via-accent/20 to-background">
      <div className="container mx-auto px-4 sm:px-6 lg:px-8 relative z-10">
        <div className="max-w-6xl mx-auto">
          <div className="text-center mb-16 animate-fade-in">
            <h2 className="text-3xl sm:text-4xl md:text-5xl font-bold mb-4 bg-gradient-to-r from-primary to-secondary bg-clip-text text-transparent">
              Get In Touch
            </h2>
            <div className="w-20 h-1 bg-gradient-to-r from-primary to-secondary mx-auto mb-8 rounded-full" />
            <p className="text-base sm:text-lg text-muted-foreground max-w-2xl mx-auto">
              Have a project in mind or want to discuss opportunities? Let's connect!
            </p>
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 lg:gap-12 items-start">
            {/* Left Side - Contact Form */}
            <div className="glass-card p-6 sm:p-8 animate-slide-up border-2 hover:border-primary/30 transition-all duration-300">
              <h3 className="text-2xl font-bold mb-6 text-foreground">Send a Message</h3>
              <form onSubmit={handleSubmit} className="space-y-4">
                <div>
                  <label htmlFor="name" className="block text-sm font-medium text-foreground mb-2">
                    Name
                  </label>
                  <Input
                    id="name"
                    value={formData.name}
                    onChange={(e) => setFormData({ ...formData, name: e.target.value })}
                    placeholder="Your name"
                    required
                    className="bg-background/50"
                  />
                </div>
                <div>
                  <label htmlFor="email" className="block text-sm font-medium text-foreground mb-2">
                    Email
                  </label>
                  <Input
                    id="email"
                    type="email"
                    value={formData.email}
                    onChange={(e) => setFormData({ ...formData, email: e.target.value })}
                    placeholder="your.email@example.com"
                    required
                    className="bg-background/50"
                  />
                </div>
                <div>
                  <label htmlFor="message" className="block text-sm font-medium text-foreground mb-2">
                    Message
                  </label>
                  <Textarea
                    id="message"
                    value={formData.message}
                    onChange={(e) => setFormData({ ...formData, message: e.target.value })}
                    placeholder="Your message..."
                    rows={5}
                    required
                    className="bg-background/50"
                  />
                </div>
                <Button
                  type="submit"
                  className="w-full bg-gradient-to-r from-primary to-secondary hover:shadow-lg hover:shadow-primary/50 transition-all duration-300"
                  size="lg"
                >
                  Send Message
                </Button>
              </form>
            </div>

            {/* Right Side - Social Links + Info */}
            <div className="space-y-6 animate-slide-up" style={{ animationDelay: "0.1s" }}>
              <div className="glass-card p-6 sm:p-8 border-2 hover:border-secondary/30 transition-all duration-300">
                <h3 className="text-2xl font-bold mb-6 text-foreground flex items-center gap-2">
                  <span className="w-2 h-2 bg-primary rounded-full animate-pulse"></span>
                  Connect With Me
                </h3>
                <div className="space-y-3">
                  {contactLinks.map((link, index) => {
                    const Icon = link.icon;
                    return (
                      <Button
                        key={link.label}
                        variant="outline"
                        className={`w-full justify-start text-left h-auto py-4 px-6 ${link.color} transition-all duration-300 hover:scale-105 hover:shadow-lg group`}
                        asChild
                        style={{ animationDelay: `${index * 0.1}s` }}
                      >
                        <a href={link.href} target="_blank" rel="noopener noreferrer" className="animate-fade-in">
                          <div className="p-2 rounded-lg bg-background/50 mr-3 group-hover:bg-primary/20 transition-colors">
                            <Icon className="h-5 w-5 flex-shrink-0" />
                          </div>
                          <div className="flex flex-col">
                            <span className="text-base font-semibold">{link.label}</span>
                            {link.label === "9347167572" && (
                              <span className="text-xs text-muted-foreground">Click to call</span>
                            )}
                          </div>
                        </a>
                      </Button>
                    );
                  })}
                </div>
              </div>

              <div className="glass-card p-6 sm:p-8 bg-gradient-to-br from-primary/10 via-secondary/10 to-primary/5 border-2 border-primary/20">
                <h3 className="text-xl font-bold mb-3 text-foreground flex items-center gap-2">
                  <span className="text-2xl">💼</span>
                  Open to Opportunities
                </h3>
                <p className="text-muted-foreground leading-relaxed">
                  I am actively seeking entry-level Data Analyst, Reporting Analyst, or BI Analyst roles and AI-ML roles. 
                  Whether it's an internship, full-time role, or exciting project collaboration,
                  I'd love to hear from you!
                </p>
              </div>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
};

export default Contact;
