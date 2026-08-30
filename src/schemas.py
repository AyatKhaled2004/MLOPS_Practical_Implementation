from pydantic import BaseModel, Field

class PricePredictionRequest(BaseModel):
    location: str = Field(..., min_length=2, description="Name of the location")
    sqft: float = Field(..., gt=100, lt=50000, description="Total square feet area (between 100 and 50000)")
    bath: int = Field(..., ge=1, le=10, description="Number of bathrooms (1 to 10)")
    bhk: int = Field(..., ge=1, le=10, description="Number of bedrooms/BHK (1 to 10)")

class PricePredictionResponse(BaseModel):
    predicted_price: float = Field(..., description="Estimated price in Lakhs")