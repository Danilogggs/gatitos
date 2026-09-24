export type Status = 'DRAFT' | 'PENDING_REVIEW' | 'PUBLISHED' | 'REJECTED' | 'ADOPTED' | 'ARCHIVED'
export type Image = { id: string; url: string; storage_path: string; position: number; is_primary: boolean }
export type Prediction = { id?: string; breed?: string | null; predicted_breed?: string | null; breed_confidence?: number | null; features?: string[]; predicted_features?: string[]; coat_pattern?: string | null; predicted_coat_pattern?: string | null; coat_confidence?: number | null; colors?: string[]; coat_length?: string | null; description?: string; model_version: string }
export type Cat = {
  id: string; name: string; status: Status; images: Image[]; features: string[]; breed: string | null;
  coat_pattern: string | null; coat_length: string | null; primary_color: string | null; secondary_color: string | null;
  sex: string | null; approximate_age: string | null; size: string | null; behavior: string | null;
  health_information: string | null; location: string | null; additional_notes: string | null;
  description: string | null; created_at: string; created_by?: string | null; prediction?: Prediction | null
}
export type Config = { ml_mode: string; features: Record<string,string>; breeds: string[] }
