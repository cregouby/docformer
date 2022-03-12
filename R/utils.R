#' Pipe operator
#'
#' See \code{magrittr::\link[magrittr:pipe]{\%>\%}} for details.
#'
#' @name %>%
#' @rdname pipe
#' @keywords internal
#' @export
#' @importFrom magrittr %>%
#' @usage lhs \%>\% rhs
#'
#' @return Returns `rhs(lhs)`.
NULL

is_file_list <- function(file) {
  # TODO turn into fs:: commands
  all(inherits(file,"character") & file.info(file)$isdir==FALSE)
}

is_path <- function(fpath) {
  # currently only a single path
  # TODO turn into fs:: commands
  all(inherits(fpath,"character") & file.info(fpath)$isdir==TRUE) & length(fpath)==1
}
