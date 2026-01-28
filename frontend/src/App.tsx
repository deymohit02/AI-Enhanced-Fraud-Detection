import { BrowserRouter, Routes, Route, Link, useLocation } from 'react-router-dom';
import LiveDashboard from './pages/LiveDashboard';
import ModelComparison from './pages/ModelComparison';

function App() {
    return (
        <BrowserRouter>
            <div className="app">
                <Header />
                <main>
                    <Routes>
                        <Route path="/" element={<LiveDashboard />} />
                        <Route path="/models" element={<ModelComparison />} />
                    </Routes>
                </main>
            </div>
        </BrowserRouter>
    );
}

function Header() {
    const location = useLocation();

    return (
        <header className="header">
            <div className="container">
                <div className="header-content">
                    <div className="logo">
                        <div className="logo-icon">⚡</div>
                        <span>FraudGuard AI</span>
                    </div>

                    <nav className="nav">
                        <Link
                            to="/"
                            className={`nav-button ${location.pathname === '/' ? 'active' : ''}`}
                        >
                            📊 Live Dashboard
                        </Link>
                        <Link
                            to="/models"
                            className={`nav-button ${location.pathname === '/models' ? 'active' : ''}`}
                        >
                            📈 Model Comparison
                        </Link>
                    </nav>
                </div>
            </div>
        </header>
    );
}

export default App;
