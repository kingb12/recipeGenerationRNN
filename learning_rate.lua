function learning_rate(lr, diff, avg_diff)
  return lr * (diff / avg_diff)
end
