package tool_test

import (
	"context"
	"fmt"
	"os"
	"testing"

	"github.com/habiliai/agentruntime/internal/tool"
	"github.com/jcooky/go-din"
	"github.com/mitchellh/mapstructure"
	"github.com/stretchr/testify/require"
)

func TestGetWeather(t *testing.T) {
	apiKey := os.Getenv("OPENWEATHER_API_KEY")
	if apiKey == "" {
		t.Skip("OPENWEATHER_API_KEY 환경 변수가 설정되지 않았습니다")
	}

	ctx := context.TODO()
	container := din.NewContainer(ctx, din.EnvTest)
	defer container.Close()

	s := din.MustGetT[tool.Manager](container)
	getWeatherTool := s.GetTool("get_weather")
	res, err := getWeatherTool.RunRaw(ctx, map[string]any{
		"location": "Seoul",
		"date":     "2023-10-01",
		"unit":     "c",
	})
	require.NoError(t, err)

	var weatherSummary tool.GetWeatherResponse
	require.NoError(t, mapstructure.Decode(res, &weatherSummary))

	t.Logf("contents: %v", weatherSummary)

	// 3. 출력
	fmt.Printf("🌡️ 최고 기온: %.2f°C\n", weatherSummary.Temperature.Max)
	fmt.Printf("🌡️ 최저 기온: %.2f°C\n", weatherSummary.Temperature.Min)
	fmt.Printf("🌡️ 오후 기온(12:00): %.2f°C\n", weatherSummary.Temperature.Afternoon)
	fmt.Printf("🌡️ 아침 기온(06:00): %.2f°C\n", weatherSummary.Temperature.Morning)
	fmt.Printf("🌡️ 저녁 기온(18:00): %.2f°C\n", weatherSummary.Temperature.Evening)
	fmt.Printf("🌡️ 밤 기온(00:00): %.2f°C\n", weatherSummary.Temperature.Night)
	fmt.Printf("💧 오후 습도: %.2f\n", weatherSummary.Humidity.Afternoon)
	fmt.Printf("🌬️ 최대 풍속: %.2fm/s (방향: %.2f°)\n", weatherSummary.Wind.Max.Speed, weatherSummary.Wind.Max.Direction)
}
