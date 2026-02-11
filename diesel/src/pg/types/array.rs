use byteorder::{NetworkEndian, ReadBytesExt, WriteBytesExt};
use core::fmt;
use std::io::Write;

use crate::deserialize::{self, FromSql, FromSqlRow};
use crate::pg::{Pg, PgTypeMetadata, PgValue};
use crate::query_builder::bind_collector::ByteWrapper;
use crate::serialize::{self, IsNull, Output, ToSql};
use crate::sql_types::{Array, HasSqlType, Nullable};

#[cfg(feature = "postgres_backend")]
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord, AsExpression, FromSqlRow)]
#[diesel(sql_type = Array<T>)]
/// Postgres allows multi-dimensional arrays of at most 6 dimensions. Internally they are stored as a flattened
/// representation with the dimension information encoded in the header. This struct represents a
/// multi-dimensional array with elements of type `T` as opposed to Vec<T> which can be used for 1d-arrays.
pub struct NdArray<T> {
    pub dims: Vec<usize>,
    pub data: Vec<T>,
}

impl<T> NdArray<T> {
    fn get(&self, indices: &[usize]) -> Option<&T> {
        let len = self.dims.len();
        if indices.len() != len {
            return None;
        }

        // formular for flat index of 3d array with shape (D, H, W) for index (d, h, w) is d*H*W + h*W + w:
        // d * (H * W) + h * W + w
        let mut flat_index = 0;
        for i in 0..len {
            if indices[i] >= self.dims[i] {
                return None;
            }
            flat_index += indices[i] * self.dims[i + 1..].iter().product::<usize>();
        }

        self.data.get(flat_index)
    }
}

#[cfg(feature = "postgres_backend")]
impl<T> HasSqlType<Array<T>> for Pg
where
    Pg: HasSqlType<T>,
{
    fn metadata(lookup: &mut Self::MetadataLookup) -> PgTypeMetadata {
        match <Pg as HasSqlType<T>>::metadata(lookup).0 {
            Ok(tpe) => PgTypeMetadata::new(tpe.array_oid, 0),
            c @ Err(_) => PgTypeMetadata(c),
        }
    }
}

#[cfg(feature = "postgres_backend")]
impl<T, ST> FromSql<Array<ST>, Pg> for Vec<T>
where
    T: FromSql<ST, Pg>,
{
    fn from_sql(value: PgValue<'_>) -> deserialize::Result<Self> {
        let mut bytes = value.as_bytes();
        let num_dimensions = bytes.read_i32::<NetworkEndian>()?;
        let has_null = bytes.read_i32::<NetworkEndian>()? != 0;
        let _oid = bytes.read_i32::<NetworkEndian>()?;

        if num_dimensions == 0 {
            return Ok(Vec::new());
        }

        let num_elements = bytes.read_i32::<NetworkEndian>()?;
        let _lower_bound = bytes.read_i32::<NetworkEndian>()?;

        if num_dimensions != 1 {
            return Err("multi-dimensional arrays are not supported".into());
        }

        (0..num_elements)
            .map(|_| {
                let elem_size = bytes.read_i32::<NetworkEndian>()?;
                if has_null && elem_size == -1 {
                    T::from_nullable_sql(None)
                } else {
                    let (elem_bytes, new_bytes) = bytes.split_at(elem_size.try_into()?);
                    bytes = new_bytes;
                    T::from_sql(PgValue::new_internal(elem_bytes, &value))
                }
            })
            .collect()
    }
}

#[cfg(feature = "postgres_backend")]
impl<T, ST> FromSql<Array<ST>, Pg> for NdArray<T>
where
    T: FromSql<ST, Pg>,
{
    fn from_sql(value: PgValue<'_>) -> deserialize::Result<Self> {
        let mut bytes = value.as_bytes();
        let num_dimensions = bytes.read_i32::<NetworkEndian>()?;
        let has_null = bytes.read_i32::<NetworkEndian>()? != 0;
        let _oid = bytes.read_i32::<NetworkEndian>()?;

        if num_dimensions == 0 {
            return Ok(NdArray {
                dims: Vec::new(),
                data: Vec::new(),
            });
        }

        if num_dimensions == 1 {
            return Err("trying to deserialize one-dimensional postgres array into NdArray<T>, use Vec<T> instead".into());
        }

        let num_dims: usize = num_dimensions
            .try_into()
            .map_err(|_| "number of dimensions must be positive")?;

        let mut dims = Vec::with_capacity(num_dims);
        let mut num_elements: i32;

        for _ in 0..num_dims {
            num_elements = bytes.read_i32::<NetworkEndian>()?;
            let _lower_bound = bytes.read_i32::<NetworkEndian>()?;

            let dim: usize = num_elements
                .try_into()
                .map_err(|_| "array dimension length must be positive")?;
            dims.push(dim);
        }

        let data = (0..dims.iter().product::<usize>())
            .map(|_| -> deserialize::Result<T> {
                let elem_size = bytes.read_i32::<NetworkEndian>()?;
                if has_null && elem_size == -1 {
                    T::from_nullable_sql(None)
                } else {
                    let (elem_bytes, new_bytes) = bytes.split_at(elem_size.try_into()?);
                    bytes = new_bytes;
                    T::from_sql(PgValue::new_internal(elem_bytes, &value))
                }
            })
            .collect::<deserialize::Result<Vec<T>>>()?;
        Ok(NdArray { dims, data })
    }
}

use crate::expression::AsExpression;
use crate::expression::bound::Bound;

macro_rules! array_as_expression {
    ($ty:ty, $sql_type:ty) => {
        #[cfg(feature = "postgres_backend")]
        // this simplifies the macro implementation
        // as some macro calls use this lifetime
        #[allow(clippy::extra_unused_lifetimes)]
        impl<'a, 'b, ST: 'static, T> AsExpression<$sql_type> for $ty {
            type Expression = Bound<$sql_type, Self>;

            fn as_expression(self) -> Self::Expression {
                Bound::new(self)
            }
        }
    };
}

array_as_expression!(&'a [T], Array<ST>);
array_as_expression!(&'a [T], Nullable<Array<ST>>);
array_as_expression!(&'a &'b [T], Array<ST>);
array_as_expression!(&'a &'b [T], Nullable<Array<ST>>);
array_as_expression!(Vec<T>, Array<ST>);
array_as_expression!(Vec<T>, Nullable<Array<ST>>);
array_as_expression!(&'a Vec<T>, Array<ST>);
array_as_expression!(&'a Vec<T>, Nullable<Array<ST>>);
array_as_expression!(&'a &'b Vec<T>, Array<ST>);
array_as_expression!(&'a &'b Vec<T>, Nullable<Array<ST>>);

#[cfg(feature = "postgres_backend")]
impl<ST, T> ToSql<Array<ST>, Pg> for [T]
where
    Pg: HasSqlType<ST>,
    T: ToSql<ST, Pg>,
{
    fn to_sql<'b>(&'b self, out: &mut Output<'b, '_, Pg>) -> serialize::Result {
        let num_dimensions = 1;
        out.write_i32::<NetworkEndian>(num_dimensions)?;
        let flags = 0;
        out.write_i32::<NetworkEndian>(flags)?;
        let element_oid = Pg::metadata(out.metadata_lookup()).oid()?;
        out.write_u32::<NetworkEndian>(element_oid)?;
        out.write_i32::<NetworkEndian>(self.len().try_into()?)?;
        let lower_bound = 1;
        out.write_i32::<NetworkEndian>(lower_bound)?;

        // This buffer is created outside of the loop to reuse the underlying memory allocation
        // For most cases all array elements will have the same serialized size
        let mut buffer = Vec::new();

        for elem in self.iter() {
            let is_null = {
                let mut temp_buffer = Output::new(ByteWrapper(&mut buffer), out.metadata_lookup());
                elem.to_sql(&mut temp_buffer)?
            };

            if let IsNull::No = is_null {
                out.write_i32::<NetworkEndian>(buffer.len().try_into()?)?;
                out.write_all(&buffer)?;
                buffer.clear();
            } else {
                // https://github.com/postgres/postgres/blob/82f8107b92c9104ec9d9465f3f6a4c6dab4c124a/src/backend/utils/adt/arrayfuncs.c#L1461
                out.write_i32::<NetworkEndian>(-1)?;
            }
        }

        Ok(IsNull::No)
    }
}

#[cfg(feature = "postgres_backend")]
impl<ST, T> ToSql<Nullable<Array<ST>>, Pg> for [T]
where
    [T]: ToSql<Array<ST>, Pg>,
    ST: 'static,
{
    fn to_sql<'b>(&'b self, out: &mut Output<'b, '_, Pg>) -> serialize::Result {
        ToSql::<Array<ST>, Pg>::to_sql(self, out)
    }
}

#[cfg(feature = "postgres_backend")]
impl<ST, T> ToSql<Array<ST>, Pg> for Vec<T>
where
    ST: 'static,
    [T]: ToSql<Array<ST>, Pg>,
    T: fmt::Debug,
{
    fn to_sql<'b>(&'b self, out: &mut Output<'b, '_, Pg>) -> serialize::Result {
        (self as &[T]).to_sql(out)
    }
}

#[cfg(feature = "postgres_backend")]
impl<ST, T> ToSql<Nullable<Array<ST>>, Pg> for Vec<T>
where
    ST: 'static,
    Vec<T>: ToSql<Array<ST>, Pg>,
{
    fn to_sql<'b>(&'b self, out: &mut Output<'b, '_, Pg>) -> serialize::Result {
        ToSql::<Array<ST>, Pg>::to_sql(self, out)
    }
}

// ---- NdArray<T> type wrappers and conversion helpers ----
#[cfg(feature = "postgres_backend")]
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord, AsExpression, FromSqlRow)]
#[diesel(sql_type = Array<T>)]
pub struct Vec2<T>(pub Vec<Vec<T>>);

#[cfg(feature = "postgres_backend")]
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord, AsExpression, FromSqlRow)]
#[diesel(sql_type = Array<T>)]
pub struct Vec3<T>(pub Vec<Vec<Vec<T>>>);

#[cfg(feature = "postgres_backend")]
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord, AsExpression, FromSqlRow)]
#[diesel(sql_type = Array<T>)]
pub struct Vec4<T>(pub Vec<Vec<Vec<Vec<T>>>>);

#[cfg(feature = "postgres_backend")]
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord, AsExpression, FromSqlRow)]
#[diesel(sql_type = Array<T>)]
pub struct Vec5<T>(pub Vec<Vec<Vec<Vec<Vec<T>>>>>);

#[cfg(feature = "postgres_backend")]
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord, AsExpression, FromSqlRow)]
#[diesel(sql_type = Array<T>)]
pub struct Vec6<T>(pub Vec<Vec<Vec<Vec<Vec<Vec<T>>>>>>);
#[cfg(feature = "postgres_backend")]
fn wrong_dim_error(expected: usize, got: usize) -> String {
    format!("expected {expected} dimensions, got {got}")
}

#[cfg(feature = "postgres_backend")]
fn shape_mismatch_error(expected: usize, got: usize) -> String {
    format!("shape mismatch: expected {expected} elements, got {got}")
}

#[cfg(feature = "postgres_backend")]
fn non_rectangular_error() -> String {
    "non-rectangular nested vector".to_string()
}

#[cfg(feature = "postgres_backend")]
impl<T> TryFrom<NdArray<T>> for Vec2<T> {
    type Error = String;

    fn try_from(nd: NdArray<T>) -> Result<Self, Self::Error> {
        if nd.dims.len() != 2 {
            return Err(wrong_dim_error(2, nd.dims.len()));
        }

        let expected_len = nd.dims.iter().product::<usize>();

        if nd.data.len() != expected_len {
            return Err(shape_mismatch_error(expected_len, nd.data.len()));
        }

        if expected_len == 0 {
            return Ok(Vec2(Vec::new()));
        }

        let d1_len = nd.dims[0];
        let d2_len = nd.dims[1];
        let mut it = nd.data.into_iter();
        let mut d1 = Vec::with_capacity(d1_len);

        for _ in 0..d1_len {
            let mut d2 = Vec::with_capacity(d2_len);
            d2.extend(it.by_ref().take(d2_len));
            d1.push(d2);
        }

        Ok(Vec2(d1))
    }
}

#[cfg(feature = "postgres_backend")]
impl<T> TryFrom<Vec2<T>> for NdArray<T> {
    type Error = String;

    fn try_from(v: Vec2<T>) -> Result<Self, Self::Error> {
        let num_rows = v.0.len();
        let num_cols = v.0.first().map(|r| r.len()).unwrap_or(0);

        // rectangular check
        if v.0.iter().any(|r| r.len() != num_cols) {
            return Err(non_rectangular_error());
        }

        let mut data = Vec::with_capacity(num_rows * num_cols);
        for mut row in v.0 {
            data.append(&mut row);
        }

        Ok(NdArray {
            dims: vec![num_rows, num_cols],
            data,
        })
    }
}

#[cfg(feature = "postgres_backend")]
impl<T> From<Vec<Vec<T>>> for Vec2<T> {
    fn from(v: Vec<Vec<T>>) -> Self {
        Vec2(v)
    }
}

#[cfg(feature = "postgres_backend")]
impl<T> From<Vec2<T>> for Vec<Vec<T>> {
    fn from(v: Vec2<T>) -> Self {
        v.0
    }
}

#[cfg(feature = "postgres_backend")]
impl<T, ST> FromSql<Array<ST>, Pg> for Vec2<T>
where
    T: FromSql<ST, Pg>,
{
    fn from_sql(value: PgValue<'_>) -> deserialize::Result<Self> {
        let nd = NdArray::<T>::from_sql(value)?;
        Vec2::try_from(nd).map_err(Into::into)
    }
}

// fn build_nested_vector<T>(curr_dim: usize, dims: &[usize]) -> Vec2<T> {
//     let curr_dim_len = dims[curr_dim];
//     if curr_dim == dims.len() - 2 {
//         let inner_dim_len = dims.last().unwrap();
//         let mut inner_vec = Vec::with_capacity(curr_dim_len);
//         for _ in 0..curr_dim_len {
//             inner_vec.push(Vec::with_capacity(*inner_dim_len));
//         }
//         return Vec2(inner_vec);
//     } else {
//         let mut current_layer = Vec::with_capacity(curr_dim_len);
//         for _ in 0..curr_dim_len {
//             current_layer.push(build_nested_vector(curr_dim + 1, dims));
//         }
//         return current_layer;
//     }
// }
// [2, 2, 3, 2]
// [
//   [
//     [
//       [1, 2], [3, 4], [5, 6]
//     ],
//     [[7, 8], [9, 10], [11, 12]]
//   ],
//   [
//     [[1, 2], [3, 4], [5, 6]],
//     [[7, 8], [9, 10], [11, 12]]
//   ]
// ]

#[cfg(feature = "postgres_backend")]
impl<T> TryFrom<NdArray<T>> for Vec3<T> {
    type Error = String;

    fn try_from(nd: NdArray<T>) -> Result<Self, Self::Error> {
        let ndims: usize = 3;

        if nd.dims.len() != ndims {
            return Err(wrong_dim_error(ndims, nd.dims.len()));
        }

        let expected_len = nd.dims.iter().product::<usize>();

        if nd.data.len() != expected_len {
            return Err(shape_mismatch_error(expected_len, nd.data.len()));
        }

        if expected_len == 0 {
            return Ok(Vec3(Vec::new()));
        }

        let d1_len = nd.dims[0];
        let d2_len = nd.dims[1];
        let d3_len = nd.dims[2];
        let mut it = nd.data.into_iter();
        let mut d1 = Vec::with_capacity(d1_len);

        for _ in 0..d1_len {
            let mut d2 = Vec::with_capacity(d2_len);
            for _ in 0..d2_len {
                let mut d3 = Vec::with_capacity(d3_len);
                d3.extend(it.by_ref().take(d3_len));
                d2.push(d3);
            }
            d1.push(d2);
        }

        Ok(Vec3(d1))
    }
}

#[cfg(feature = "postgres_backend")]
impl<T> TryFrom<Vec3<T>> for NdArray<T> {
    type Error = String;

    fn try_from(v: Vec3<T>) -> Result<Self, Self::Error> {
        let d1_len = v.0.len();
        let d2_len = v.0.first().map(|d2| d2.len()).unwrap_or(0);
        let d3_len =
            v.0.first()
                .map(|d2| d2.first().map(|d3| d3.len()).unwrap_or(0))
                .unwrap_or(0);

        let data_len = d1_len * d2_len * d3_len;
        if data_len == 0 {
            return Ok(NdArray {
                dims: vec![d1_len, d2_len, d3_len],
                data: Vec::new(),
            });
        }

        // rectangular check
        if v.0.iter().any(|d2| d2.len() != d2_len) {
            return Err(non_rectangular_error());
        }
        if v.0.iter().any(|d2| d2.iter().any(|d3| d3.len() != d3_len)) {
            return Err(non_rectangular_error());
        }

        let mut data = Vec::with_capacity(data_len);
        for d2 in v.0 {
            for d3 in d2 {
                data.extend(d3);
            }
        }

        Ok(NdArray {
            dims: vec![d1_len, d2_len, d3_len],
            data,
        })
    }
}

#[cfg(feature = "postgres_backend")]
impl<T> From<Vec<Vec<Vec<T>>>> for Vec3<T> {
    fn from(v: Vec<Vec<Vec<T>>>) -> Self {
        Vec3(v)
    }
}

#[cfg(feature = "postgres_backend")]
impl<T> From<Vec3<T>> for Vec<Vec<Vec<T>>> {
    fn from(v: Vec3<T>) -> Self {
        v.0
    }
}

#[cfg(feature = "postgres_backend")]
impl<T, ST> FromSql<Array<ST>, Pg> for Vec3<T>
where
    T: FromSql<ST, Pg>,
{
    fn from_sql(value: PgValue<'_>) -> deserialize::Result<Self> {
        let nd = NdArray::<T>::from_sql(value)?;
        Vec3::try_from(nd).map_err(Into::into)
    }
}

// ---- tests ----
// TODO: add tests for edge cases, like zero-length dimensions, zero-length data and NULL values
#[cfg(test)]
mod tests {
    use super::*;

    #[diesel_test_helper::test]
    fn vec2_to_ndarray() {
        let v = Vec2(vec![vec![1, 2, 3, 4, 5, 6]]);
        let nd = NdArray::try_from(v).unwrap();

        assert_eq!(nd.dims, vec![1, 6]);
        assert_eq!(nd.data, vec![1, 2, 3, 4, 5, 6]);

        let v = Vec2(vec![vec![1, 2, 3], vec![4, 5, 6]]);
        let nd = NdArray::try_from(v).unwrap();

        assert_eq!(nd.dims, vec![2, 3]);
        assert_eq!(nd.data, vec![1, 2, 3, 4, 5, 6]);
    }

    #[diesel_test_helper::test]
    fn ndarray_to_vec2() {
        let nd = NdArray {
            dims: vec![2, 3],
            data: vec![1, 2, 3, 4, 5, 6],
        };
        let v = Vec2::try_from(nd).unwrap();

        assert_eq!(v.0, vec![vec![1, 2, 3], vec![4, 5, 6]]);

        let nd = NdArray {
            dims: vec![1, 6],
            data: vec![1, 2, 3, 4, 5, 6],
        };
        let v = Vec2::try_from(nd).unwrap();

        assert_eq!(v.0, vec![vec![1, 2, 3, 4, 5, 6]]);
    }

    #[diesel_test_helper::test]
    fn bad_ndarray_to_vec2() {
        let nd = NdArray {
            dims: vec![1, 2],
            data: vec![1, 2, 3],
        };
        let err = Vec2::try_from(nd).unwrap_err();

        assert_eq!(err.to_string(), shape_mismatch_error(2, 3));

        let nd = NdArray {
            dims: vec![1, 3, 1],
            data: vec![vec![vec![1], vec![2], vec![3]]],
        };
        let err = Vec2::try_from(nd).unwrap_err();

        assert_eq!(err.to_string(), wrong_dim_error(2, 3));
    }

    #[diesel_test_helper::test]
    fn jagged_vec2_to_ndarray() {
        let v = Vec2(vec![vec![1], vec![2, 3]]);
        let err = NdArray::try_from(v).unwrap_err();

        assert_eq!(err.to_string(), non_rectangular_error());
    }

    #[diesel_test_helper::test]
    fn vec3_to_ndarray() {
        let v = Vec3(vec![vec![vec![1, 2, 3, 4, 5, 6]]]);
        let nd = NdArray::try_from(v).unwrap();
        assert_eq!(nd.dims, vec![1, 1, 6]);
        assert_eq!(nd.data, vec![1, 2, 3, 4, 5, 6]);

        let v = Vec3(vec![vec![vec![1, 2, 3], vec![4, 5, 6]]]);
        let nd = NdArray::try_from(v).unwrap();
        assert_eq!(nd.dims, vec![1, 2, 3]);
        assert_eq!(nd.data, vec![1, 2, 3, 4, 5, 6]);

        let v = Vec3(vec![vec![vec![1, 2], vec![2, 3], vec![3, 4]]]);
        let nd = NdArray::try_from(v).unwrap();
        assert_eq!(nd.dims, vec![1, 3, 2]);
        assert_eq!(nd.data, vec![1, 2, 2, 3, 3, 4]);
    }

    #[diesel_test_helper::test]
    fn ndarray_to_vec3() {
        let nd = NdArray {
            dims: vec![1, 2, 3],
            data: vec![1, 2, 3, 4, 5, 6],
        };
        let v = Vec3::try_from(nd).unwrap();
        assert_eq!(v.0, vec![vec![vec![1, 2, 3], vec![4, 5, 6]]]);

        let nd = NdArray {
            dims: vec![1, 6, 1],
            data: vec![1, 2, 3, 4, 5, 6],
        };
        let v = Vec3::try_from(nd).unwrap();
        assert_eq!(
            v.0,
            vec![vec![vec![1], vec![2], vec![3], vec![4], vec![5], vec![6]]]
        );
    }

    #[diesel_test_helper::test]
    fn bad_ndarray_to_vec3() {
        let nd = NdArray {
            dims: vec![1, 2, 1],
            data: vec![1, 2, 3],
        };
        let err = Vec3::try_from(nd).unwrap_err();

        assert_eq!(err.to_string(), shape_mismatch_error(2, 3));

        let nd = NdArray {
            dims: vec![1, 3],
            data: vec![vec![vec![1], vec![2]]],
        };
        let err = Vec3::try_from(nd).unwrap_err();

        assert_eq!(err.to_string(), wrong_dim_error(3, 2));
    }

    #[diesel_test_helper::test]
    fn jagged_vec3_to_ndarray() {
        let v = Vec3(vec![vec![vec![1]], vec![vec![2, 3]]]);
        let err = NdArray::try_from(v).unwrap_err();

        assert_eq!(err.to_string(), non_rectangular_error());
    }

    #[diesel_test_helper::test]
    fn test_ndarray_get_2d() {
        let nd = NdArray {
            dims: vec![2, 3],
            data: vec![1, 2, 3, 4, 5, 6],
        };

        assert_eq!(nd.get(&[0, 0]), Some(&1));
        assert_eq!(nd.get(&[0, 1]), Some(&2));
        assert_eq!(nd.get(&[0, 2]), Some(&3));
        assert_eq!(nd.get(&[1, 0]), Some(&4));
        assert_eq!(nd.get(&[1, 1]), Some(&5));
        assert_eq!(nd.get(&[1, 2]), Some(&6));

        // out of bounds
        assert_eq!(nd.get(&[2, 0]), None);
        assert_eq!(nd.get(&[0, 3]), None);

        // wrong number of indices
        assert_eq!(nd.get(&[0]), None);
        assert_eq!(nd.get(&[0, 0, 0]), None);
    }

    #[diesel_test_helper::test]
    fn test_ndarray_get_3d() {
        let nd = NdArray {
            dims: vec![2, 3, 2],
            data: vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
        };

        assert_eq!(nd.get(&[0, 0, 0]), Some(&1));
        assert_eq!(nd.get(&[0, 0, 1]), Some(&2));
        assert_eq!(nd.get(&[0, 1, 1]), Some(&4));
        assert_eq!(nd.get(&[0, 2, 0]), Some(&5));
        assert_eq!(nd.get(&[1, 0, 0]), Some(&7));
        assert_eq!(nd.get(&[1, 2, 1]), Some(&12));

        // out of bounds
        assert_eq!(nd.get(&[2, 0, 1]), None);
        assert_eq!(nd.get(&[0, 3, 1]), None);

        // wrong number of indices
        assert_eq!(nd.get(&[0, 0]), None);
    }

    #[diesel_test_helper::test]
    fn test_ndarray_get_5d() {
        let nd = NdArray {
            dims: vec![1, 3, 2, 2, 1],
            data: vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
        };

        // [
        //   [
        //     [[1], [2]],
        //     [[3], [4]]],
        //   [
        //     [[5], [6]],
        //     [[7], [8]]],
        //   [
        //     [[9], [10]],
        //     [[11], [12]]],
        // ]

        assert_eq!(nd.get(&[0, 0, 0, 1, 0]), Some(&2));
        assert_eq!(nd.get(&[0, 0, 1, 1, 0]), Some(&4));
        assert_eq!(nd.get(&[0, 2, 1, 0, 0]), Some(&11));

        // out of bounds
        assert_eq!(nd.get(&[2, 0, 1, 0, 0]), None);
        assert_eq!(nd.get(&[0, 3, 1, 1, 0]), None);

        // wrong number of indices
        assert_eq!(nd.get(&[0, 0]), None);
        assert_eq!(nd.get(&[0, 0, 0]), None);
        assert_eq!(nd.get(&[0, 0, 0, 0, 0, 0]), None);
    }
}
