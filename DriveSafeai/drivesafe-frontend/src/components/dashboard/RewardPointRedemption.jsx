import { useState, useEffect } from "react";
import { Plus, Star, Gift, Sparkles } from "lucide-react";
import UserService from "../../services/UserService";

const RewardsPage = () => {
    const [userPoints, setUserPoints] = useState(0);
    const [selectedCards, setSelectedCards] = useState({});

    const rewardCards = [
        {
            id: 1,
            brand: "Burger King",
            points: 1000,
            value: "₹50",
            description: "Whopper Burger Combo",
            color: "#ff6b6b",
            icon: "🍔"
        },
        {
            id: 2,
            brand: "Indian Oil",
            points: 2000,
            value: "₹100",
            description: "Fuel Voucher",
            color: "#4dabf7",
            icon: "⛽"
        },
        {
            id: 3,
            brand: "Swiggy",
            points: 1500,
            value: "₹75",
            description: "Food Delivery Discount",
            color: "#ff8cc8",
            icon: "🍽️"
        },
        {
            id: 4,
            brand: "Amazon",
            points: 2500,
            value: "₹125",
            description: "Shopping Voucher",
            color: "#ffd43b",
            icon: "📦"
        },
        {
            id: 5,
            brand: "Netflix",
            points: 3000,
            value: "₹150",
            description: "1 Month Subscription",
            color: "#e03131",
            icon: "🎬"
        },
        {
            id: 6,
            brand: "Spotify",
            points: 1800,
            value: "₹90",
            description: "Premium Music Access",
            color: "#51cf66",
            icon: "🎵"
        },
        {
            id: 7,
            brand: "Zomato",
            points: 1200,
            value: "₹60",
            description: "Food Order Discount",
            color: "#9775fa",
            icon: "🍕"
        }
    ];

    useEffect(() => {
        const fetchPoints = async () => {
            const userId = localStorage.getItem('userId');
            const points = await UserService.getRewardPoints(userId);
            setUserPoints(points);
        };
        fetchPoints();
    }, []);

    const handleAddReward = (card) => {
        if (userPoints >= card.points) {
            setSelectedCards(prev => ({
                ...prev,
                [card.id]: (prev[card.id] || 0) + 1
            }));
        }
    };

    const canAfford = (points) => userPoints >= points;

    return (
        <div style={{
            minHeight: '100vh',
            background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
            fontFamily: 'system-ui, -apple-system, sans-serif'
        }}>
            <style jsx>{`
                .card-hover {
                    transition: transform 0.3s ease, box-shadow 0.3s ease;
                }
                .card-hover:hover {
                    transform: translateY(-8px);
                    box-shadow: 0 20px 40px rgba(0,0,0,0.2);
                }
                .button-hover {
                    transition: all 0.2s ease;
                }
                .button-hover:hover {
                    transform: scale(1.02);
                }
                .button-hover:active {
                    transform: scale(0.98);
                }
                .fade-in {
                    animation: fadeIn 0.6s ease-out;
                }
                @keyframes fadeIn {
                    from { opacity: 0; transform: translateY(20px); }
                    to { opacity: 1; transform: translateY(0); }
                }
                .glass {
                    backdrop-filter: blur(10px);
                    background: rgba(255, 255, 255, 0.1);
                    border: 1px solid rgba(255, 255, 255, 0.2);
                }
            `}</style>

            {/* Header */}
            <div className="glass" style={{ position: 'sticky', top: 0, zIndex: 10 }}>
                <div style={{ maxWidth: '1200px', margin: '0 auto', padding: '20px' }}>
                    <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', flexWrap: 'wrap', gap: '16px' }}>
                        <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
                            <div style={{
                                width: '44px',
                                height: '44px',
                                background: 'linear-gradient(135deg, #ff6b6b, #4ecdc4)',
                                borderRadius: '12px',
                                display: 'flex',
                                alignItems: 'center',
                                justifyContent: 'center'
                            }}>
                                <Gift style={{ width: '24px', height: '24px', color: 'white' }} />
                            </div>
                            <div>
                                <h1 style={{
                                    fontSize: '28px',
                                    fontWeight: '700',
                                    color: 'white',
                                    margin: 0,
                                    textShadow: '0 2px 4px rgba(0,0,0,0.3)'
                                }}>
                                    Rewards Store
                                </h1>
                                <p style={{ color: 'rgba(255,255,255,0.8)', fontSize: '14px', margin: 0 }}>
                                    Redeem your points for amazing offers
                                </p>
                            </div>
                        </div>
                        <div style={{
                            display: 'flex',
                            alignItems: 'center',
                            gap: '8px',
                            background: 'rgba(255,255,255,0.15)',
                            padding: '12px 20px',
                            borderRadius: '25px',
                            border: '1px solid rgba(255,255,255,0.3)'
                        }}>
                            <Sparkles style={{ width: '18px', height: '18px', color: '#ffd700' }} />
                            <span style={{ fontWeight: '600', color: 'white', fontSize: '16px' }}>
                {userPoints.toLocaleString()} Points
              </span>
                        </div>
                    </div>
                </div>
            </div>

            {/* Main Content */}
            <div style={{ maxWidth: '1200px', margin: '0 auto', padding: '40px 20px' }}>
                <div style={{ marginBottom: '40px', textAlign: 'center' }}>
                    <h2 style={{
                        fontSize: '36px',
                        fontWeight: '700',
                        color: 'white',
                        marginBottom: '12px',
                        textShadow: '0 2px 8px rgba(0,0,0,0.3)'
                    }}>
                        Popular Rewards
                    </h2>
                    <p style={{ color: 'rgba(255,255,255,0.9)', fontSize: '16px', margin: 0 }}>
                        Choose from our selection of premium rewards and offers
                    </p>
                </div>

                {/* Cards Grid */}
                <div style={{
                    display: 'grid',
                    gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))',
                    gap: '24px',
                    marginBottom: '40px'
                }}>
                    {rewardCards.map((card) => (
                        <div
                            key={card.id}
                            className="card-hover fade-in"
                            style={{
                                background: 'rgba(255, 255, 255, 0.95)',
                                borderRadius: '16px',
                                padding: '24px',
                                position: 'relative',
                                opacity: canAfford(card.points) ? 1 : 0.6,
                                boxShadow: '0 8px 32px rgba(0,0,0,0.1)'
                            }}
                        >
                            {/* Card Header */}
                            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '20px' }}>
                                <div style={{ fontSize: '36px' }}>{card.icon}</div>
                                {selectedCards[card.id] && (
                                    <div style={{
                                        background: card.color,
                                        color: 'white',
                                        fontSize: '12px',
                                        padding: '4px 12px',
                                        borderRadius: '12px',
                                        fontWeight: '600'
                                    }}>
                                        {selectedCards[card.id]} Added
                                    </div>
                                )}
                            </div>

                            {/* Brand Name */}
                            <h3 style={{
                                fontWeight: '700',
                                fontSize: '20px',
                                color: '#2d3748',
                                marginBottom: '8px',
                                margin: '0 0 8px 0'
                            }}>
                                {card.brand}
                            </h3>

                            {/* Description */}
                            <p style={{
                                color: '#718096',
                                fontSize: '14px',
                                marginBottom: '16px',
                                margin: '0 0 16px 0',
                                lineHeight: '1.4'
                            }}>
                                {card.description}
                            </p>

                            {/* Value */}
                            <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '16px' }}>
                <span style={{
                    fontSize: '24px',
                    fontWeight: '700',
                    color: card.color
                }}>
                  {card.value}
                </span>
                                <span style={{ fontSize: '14px', color: '#a0aec0' }}>value</span>
                            </div>

                            {/* Points Required */}
                            <div style={{ display: 'flex', alignItems: 'center', gap: '6px', marginBottom: '20px' }}>
                                <Star style={{ width: '16px', height: '16px', color: '#ffd700', fill: '#ffd700' }} />
                                <span style={{ fontWeight: '600', color: '#4a5568', fontSize: '14px' }}>
                  {card.points.toLocaleString()} points
                </span>
                            </div>

                            {/* Add Button */}
                            <button
                                onClick={() => handleAddReward(card)}
                                disabled={!canAfford(card.points)}
                                className="button-hover"
                                style={{
                                    width: '100%',
                                    display: 'flex',
                                    alignItems: 'center',
                                    justifyContent: 'center',
                                    gap: '8px',
                                    padding: '12px 16px',
                                    borderRadius: '12px',
                                    fontWeight: '600',
                                    fontSize: '14px',
                                    border: 'none',
                                    cursor: canAfford(card.points) ? 'pointer' : 'not-allowed',
                                    background: canAfford(card.points) ? card.color : '#e2e8f0',
                                    color: canAfford(card.points) ? 'white' : '#a0aec0'
                                }}
                            >
                                <Plus style={{ width: '18px', height: '18px' }} />
                                {canAfford(card.points) ? 'Add Reward' : 'Insufficient Points'}
                            </button>
                        </div>
                    ))}
                </div>

                {/* Selected Rewards Summary */}
                {Object.keys(selectedCards).length > 0 && (
                    <div className="glass fade-in" style={{
                        borderRadius: '16px',
                        padding: '24px',
                        border: '1px solid rgba(255,255,255,0.2)'
                    }}>
                        <h3 style={{
                            fontSize: '20px',
                            fontWeight: '700',
                            color: 'white',
                            marginBottom: '20px',
                            display: 'flex',
                            alignItems: 'center',
                            gap: '8px',
                            margin: '0 0 20px 0'
                        }}>
                            <Gift style={{ width: '22px', height: '22px', color: '#ffd700' }} />
                            Selected Rewards
                        </h3>
                        <div style={{
                            display: 'grid',
                            gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))',
                            gap: '12px'
                        }}>
                            {Object.entries(selectedCards).map(([cardId, count]) => {
                                const card = rewardCards.find(c => c.id === parseInt(cardId));
                                return (
                                    <div
                                        key={cardId}
                                        style={{
                                            display: 'flex',
                                            alignItems: 'center',
                                            justifyContent: 'space-between',
                                            background: 'rgba(255,255,255,0.1)',
                                            padding: '12px 16px',
                                            borderRadius: '12px',
                                            border: '1px solid rgba(255,255,255,0.2)'
                                        }}
                                    >
                                        <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                                            <span style={{ fontSize: '18px' }}>{card.icon}</span>
                                            <span style={{ fontWeight: '500', color: 'white', fontSize: '14px' }}>{card.brand}</span>
                                        </div>
                                        <div style={{
                                            background: card.color,
                                            color: 'white',
                                            padding: '4px 8px',
                                            borderRadius: '8px',
                                            fontSize: '12px',
                                            fontWeight: '600'
                                        }}>
                                            {count}x
                                        </div>
                                    </div>
                                );
                            })}
                        </div>
                    </div>
                )}
            </div>
        </div>
    );
};

export default RewardsPage;